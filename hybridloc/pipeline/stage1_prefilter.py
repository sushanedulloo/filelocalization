"""Stage 1: lightweight pre-filter.

Three retrievers (BM25, dense, LLM) → union of top-K → cap at 20 files.
Recall@20 target: 0.85 on the 30-issue dev split.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.nim_client import NIMClient, ReasoningMode
from ..parsing.skeleton import Skeleton, build_repo_skeleton, load_skeletons, save_skeletons
from ..retrieval.dense import DenseRetriever
from ..retrieval.sparse import BM25Retriever


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

        # 1) BM25
        bm25 = BM25Retriever(texts, ids)
        bm25_top = bm25.query(issue, top_k=self.per_retriever_k)

        # 2) Dense (optional — skipped if no model wired or HYBRIDLOC_SKIP_DENSE=1)
        import os
        dense_top: list[tuple[str, float]] = []
        if self.dense is not None and os.environ.get("HYBRIDLOC_SKIP_DENSE", "0") != "1":
            try:
                self.dense.index(texts, ids)
                dense_top = self.dense.query(issue, top_k=self.per_retriever_k)
            except Exception as e:
                notes.append(f"dense retriever failed: {e!r}")

        # 3) LLM rank (optional — skipped if no NIM client)
        llm_top: list[tuple[str, str]] = []
        if self.nim is not None:
            try:
                llm_top = self._llm_rank(issue, skels, notes)
            except Exception as e:
                notes.append(f"llm rank failed: {e!r}")

        # Research-backed merge (Agentless / LocAgent style):
        # BM25 and Dense are *pre-filters* that widen the candidate pool for
        # recall. The LLM is the *primary ranker* — its order is the final order.
        # Files found only by BM25/Dense (not by LLM) are appended at the end so
        # they still appear in top-K for recall, but never override LLM ordering.
        seen: set[str] = set()
        merged: list[str] = []
        # 1) LLM order first (primary ranker)
        for p, _ in llm_top:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        # 2) BM25 results not already included (recall booster)
        for p, _ in bm25_top:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        # 3) Dense results not already included
        for p, _ in dense_top:
            if p not in seen:
                seen.add(p)
                merged.append(p)
        # Fallback: if LLM call failed and merged is empty, fall back to BM25
        if not merged:
            for p, _ in bm25_top + dense_top:
                if p not in seen:
                    seen.add(p)
                    merged.append(p)

        from ..log import info
        result = Stage1Result(
            candidate_files=merged[: self.top_k],
            bm25_top=bm25_top,
            dense_top=dense_top,
            llm_top=llm_top,
            skeleton_count=len(skels),
            notes=notes,
        )
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
