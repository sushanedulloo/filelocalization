"""Stage 1: lightweight pre-filter.

Architecture (size-adaptive):
  Small/medium repos (outline fits): BM25 + Dense + LLM run independently,
    merged via RRF (Reciprocal Rank Fusion). LLM gets a higher weight (1.0)
    than BM25/Dense (0.7) but cannot single-handedly override agreement
    between the other two.
  Large repos (outline would be truncated): BM25 + Dense act as Agentless-
    style pre-filters → LLM only ranks the filtered pool of ~50 files →
    LLM order is the final order (no truncation possible).

Test files get a 0.5x soft penalty in the final scoring so they don't
dominate implementation files (LocAgent / RepoLens convention).

Recall@20 target: 0.85 on the 30-issue dev split.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..graph.seeds import _is_test_file
from ..llm.nim_client import NIMClient, ReasoningMode
from ..parsing.skeleton import Skeleton, build_repo_skeleton, load_skeletons, save_skeletons
from ..retrieval.dense import DenseRetriever
from ..retrieval.sparse import BM25Retriever


# Reciprocal Rank Fusion constant (Cormack et al. 2009). k=60 is the
# standard value used across most modern multi-retriever RAG systems.
_RRF_K = 60

# Per-retriever weight in RRF. LLM understands semantics → higher weight.
# BM25 and Dense are keyword/embedding matchers → still useful but less.
_RRF_WEIGHT_LLM = 1.0
_RRF_WEIGHT_BM25 = 0.7
_RRF_WEIGHT_DENSE = 0.7

# Soft penalty for test files in the final merged score.
_TEST_FILE_PENALTY = 0.5


@dataclass
class Stage1Result:
    candidate_files: list[str]
    bm25_top: list[tuple[str, float]]
    dense_top: list[tuple[str, float]]
    llm_top: list[tuple[str, str]]   # (path, reason)
    skeleton_count: int
    notes: list[str] = field(default_factory=list)


class PreFilter:
    def __init__(
        self,
        nim: NIMClient | None = None,
        dense: DenseRetriever | None = None,
        *,
        top_k: int = 20,
        per_retriever_k: int = 20,
        llm_top_k: int = 10,
        llm_max_outline_chars: int = 600_000,
    ):
        self.nim = nim
        self.dense = dense
        self.top_k = top_k
        self.per_retriever_k = per_retriever_k
        self.llm_top_k = llm_top_k
        self.llm_max_outline_chars = llm_max_outline_chars
        self._prompt_template = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "prompts"
            / "file_rank.txt"
        ).read_text()

    # ---------- offline indexing ----------

    @staticmethod
    def build_or_load_index(repo_root: Path, cache_path: Path) -> list[Skeleton]:
        if cache_path.exists():
            return load_skeletons(cache_path)
        skels = build_repo_skeleton(repo_root)
        save_skeletons(skels, cache_path)
        return skels

    # ---------- per-issue ----------

    def run(self, issue: str, skels: list[Skeleton]) -> Stage1Result:
        if not skels:
            return Stage1Result([], [], [], [], 0, notes=["empty skeleton index"])

        notes: list[str] = []
        ids = [s.file_path for s in skels]
        texts = [s.as_compact_text() for s in skels]

        # Decide architecture based on repo size FIRST, so we can request
        # bigger retriever top-K when pre-filter mode is needed.
        # Threshold = 5x llm_max_outline_chars (default 3M chars). Below
        # this, truncation still leaves a usable fraction of files visible
        # to the LLM. Django at 1.8M used fusion mode successfully — the
        # correct file was in the 614 visible files. Pre-filter only kicks
        # in for genuinely huge repos where truncation drops most files.
        total_outline = sum(len(s.as_compact_text()) + 2 for s in skels)
        large_repo = total_outline > 5 * self.llm_max_outline_chars

        # In pre-filter mode we need a wider BM25/Dense pool so BM25's misses
        # don't cost us the gold file. In fusion mode the LLM sees the full
        # outline anyway, so the smaller per_retriever_k is fine.
        retriever_k = 100 if large_repo else self.per_retriever_k

        # 1) BM25
        bm25 = BM25Retriever(texts, ids)
        bm25_top = bm25.query(issue, top_k=retriever_k)

        # 2) Dense (optional — skipped if no model wired or HYBRIDLOC_SKIP_DENSE=1)
        import os
        dense_top: list[tuple[str, float]] = []
        if self.dense is not None and os.environ.get("HYBRIDLOC_SKIP_DENSE", "0") != "1":
            try:
                self.dense.index(texts, ids)
                dense_top = self.dense.query(issue, top_k=retriever_k)
            except Exception as e:
                notes.append(f"dense retriever failed: {e!r}")

        llm_top: list[tuple[str, str]] = []
        if large_repo and self.nim is not None:
            # PRE-FILTER MODE — BM25 ∪ Dense → ~150 candidates → LLM ranks those
            pool_paths: list[str] = []
            seen: set[str] = set()
            for src in (bm25_top, dense_top):
                for p, _ in src[:retriever_k]:
                    if p not in seen:
                        seen.add(p)
                        pool_paths.append(p)
            pool_skels = [s for s in skels if s.file_path in seen]
            notes.append(
                f"large-repo mode: BM25+Dense pre-filter reduced {len(skels)} → "
                f"{len(pool_skels)} files for LLM ranking"
            )
            try:
                llm_top = self._llm_rank(issue, pool_skels, notes)
            except Exception as e:
                notes.append(f"llm rank failed: {e!r}")
            # In pre-filter mode the LLM is the only ranker over the pool, so
            # its order IS the final order. RRF would over-weight BM25/Dense
            # signals that the LLM already considered.
            merged_scored = [
                (p, 1.0 / (_RRF_K + rank))
                for rank, (p, _) in enumerate(llm_top)
            ]
        else:
            # FUSION MODE — all three retrievers independent, RRF merge
            if self.nim is not None:
                try:
                    llm_top = self._llm_rank(issue, skels, notes)
                except Exception as e:
                    notes.append(f"llm rank failed: {e!r}")
            merged_scored = _rrf_merge(llm_top, bm25_top, dense_top)

        # Apply soft test-file penalty
        final_scored = [
            (p, s * (_TEST_FILE_PENALTY if _is_test_file(p) else 1.0))
            for p, s in merged_scored
        ]
        final_scored.sort(key=lambda x: -x[1])
        merged = [p for p, _ in final_scored]

        # Fallback: if everything failed, use BM25 alone
        if not merged:
            merged = [p for p, _ in bm25_top + dense_top]

        from ..log import info
        result = Stage1Result(
            candidate_files=merged[: self.top_k],
            bm25_top=bm25_top,
            dense_top=dense_top,
            llm_top=llm_top,
            skeleton_count=len(skels),
            notes=notes,
        )
        info(f"[Stage 1] mode={'pre-filter' if large_repo else 'fusion-rrf'}  total_files={len(skels)}  outline_chars={total_outline}")
        info(f"[Stage 1] BM25 top-5: {[p for p, _ in bm25_top[:5]]}")
        info(f"[Stage 1] Dense top-5: {[p for p, _ in dense_top[:5]]}")
        info(f"[Stage 1] LLM top-5:  {[p for p, r in llm_top[:5]]}")
        for p, r in llm_top[:5]:
            info(f"[Stage 1]   → {p}: {r[:120]}")
        info(f"[Stage 1] Final merged top-10: {merged[:10]}")
        if notes:
            for n in notes:
                info(f"[Stage 1] NOTE: {n}")
        return result

    # ---------- internals ----------

    def _llm_rank(
        self, issue: str, skels: list[Skeleton], notes: list[str]
    ) -> list[tuple[str, str]]:
        outline = self._build_outline(skels, self.llm_max_outline_chars, notes)
        prompt = self._prompt_template.format(
            issue=issue.strip()[:8000],
            outline=outline,
            top_k=self.llm_top_k,
        )
        resp = self.nim.complete(  # type: ignore[union-attr]
            prompt,
            mode=ReasoningMode.NON_THINK,
            json_schema={"type": "object"},
            temperature=0.0,
        )
        return _parse_llm_rank(resp.text)

    @staticmethod
    def _build_outline(
        skels: list[Skeleton], max_chars: int, notes: list[str]
    ) -> str:
        chunks: list[str] = []
        total = 0
        for s in skels:
            block = s.as_compact_text()
            if total + len(block) + 2 > max_chars:
                notes.append(
                    f"outline truncated at {total} chars; "
                    f"{len(skels) - len(chunks)} files dropped from LLM ranker"
                )
                break
            chunks.append(block)
            total += len(block) + 2
        return "\n\n".join(chunks)


def _rrf_merge(
    llm_top: list[tuple[str, str]],
    bm25_top: list[tuple[str, float]],
    dense_top: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion of multiple ranked lists.

    score(file) = sum over retrievers of weight_r / (k + rank_r(file))

    Cormack et al. 2009. Each retriever contributes proportional to where it
    ranks the file. A file ranked highly by multiple retrievers wins. A file
    ranked only by one retriever still surfaces but at a lower score, so
    BM25/Dense recall hits are preserved while LLM-confirmed files dominate.
    """
    scores: dict[str, float] = {}

    def _add(ranked: list, weight: float) -> None:
        for rank, item in enumerate(ranked):
            path = item[0]
            scores[path] = scores.get(path, 0.0) + weight / (_RRF_K + rank)

    _add(llm_top, _RRF_WEIGHT_LLM)
    _add(bm25_top, _RRF_WEIGHT_BM25)
    _add(dense_top, _RRF_WEIGHT_DENSE)

    return sorted(scores.items(), key=lambda x: -x[1])


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _parse_llm_rank(raw: str) -> list[tuple[str, str]]:
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    files = obj.get("files") if isinstance(obj, dict) else None
    if not isinstance(files, list):
        return []
    out: list[tuple[str, str]] = []
    for entry in files:
        if isinstance(entry, dict) and "path" in entry:
            out.append((str(entry["path"]), str(entry.get("reason", ""))))
    return out
