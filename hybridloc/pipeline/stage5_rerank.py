"""Stage 5: bi-encoder retrieval + RGFL explanation re-embedding + listwise LLM rerank."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from ..graph.nodes import NodeType
from ..llm.nim_client import NIMClient, ReasoningMode
from ..pipeline.stage4_traversal import Candidate
from ..retrieval.dense import DenseRetriever

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


@dataclass
class RankedItem:
    node_id: str
    function_key: str          # "<file_path>::<qualname>"
    score: float
    confidence: str            # "high" | "medium" | "low"
    causal_chain: list[str] = field(default_factory=list)
    suspect_lines: tuple[int, int] | None = None
    file_path: str = ""
    qualname: str = ""


class Reranker:
    def __init__(
        self,
        nim: NIMClient,
        dense: DenseRetriever | None,
        *,
        listwise_top_k: int = 10,
        retrieval_top_k: int = 20,
    ):
        self.nim = nim
        self.dense = dense
        self.listwise_top_k = listwise_top_k
        self.retrieval_top_k = retrieval_top_k
        self._listwise_tpl = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "prompts"
            / "listwise_rerank.txt"
        ).read_text()

    def rerank(
        self,
        *,
        graph: nx.MultiDiGraph,
        issue: str,
        candidates: list[Candidate],
    ) -> list[RankedItem]:
        if not candidates:
            return []

        # 1. Bi-encoder rescore (issue + chain) vs candidate function code
        scored = self._embedding_rerank(graph, issue, candidates)
        top = scored[: self.retrieval_top_k]

        # 2. Listwise LLM rerank on top-K
        if len(top) > 1:
            top = self._listwise_rerank(graph, issue, top)

        # 3. confidence labels (set by consistency-vote step downstream; defaults here)
        out: list[RankedItem] = []
        for nid, score, cand in top:
            data = graph.nodes[nid].get("data")
            if data is None:
                continue
            suspect_lines = self._suspect_lines(graph, cand)
            out.append(
                RankedItem(
                    node_id=nid,
                    function_key=f"{data.file_path}::{data.qualname}",
                    score=float(score),
                    confidence="medium",
                    causal_chain=cand.causal_chain,
                    suspect_lines=suspect_lines,
                    file_path=data.file_path,
                    qualname=data.qualname,
                )
            )
        return out

    # ---------------- internals ----------------

    def _embedding_rerank(
        self, graph: nx.MultiDiGraph, issue: str, candidates: list[Candidate]
    ) -> list[tuple[str, float, Candidate]]:
        if self.dense is None:
            # fall back to traversal score order
            return [(c.node_id, c.score, c) for c in candidates]
        # build augmented query: issue + concatenated chains (RGFL re-embedding)
        chain_text = " ".join(
            " ".join(c.causal_chain) for c in candidates if c.causal_chain
        )[:2000]
        q_text = issue + ("\n" + chain_text if chain_text else "")
        q_emb = self.dense.encode([q_text])[0]

        cand_texts: list[str] = []
        for c in candidates:
            data = graph.nodes[c.node_id].get("data")
            if data is None:
                cand_texts.append("")
                continue
            cand_texts.append(
                f"{data.qualname}\n{data.docstring}\n{data.code or ''}"[:4000]
            )
        c_embs = self.dense.encode(cand_texts) if cand_texts else np.zeros((0,))
        if c_embs.size == 0:
            return [(c.node_id, c.score, c) for c in candidates]
        sims = c_embs @ q_emb
        order = np.argsort(-sims)
        return [(candidates[i].node_id, float(sims[i]), candidates[i]) for i in order]

    def _listwise_rerank(
        self,
        graph: nx.MultiDiGraph,
        issue: str,
        top: list[tuple[str, float, Candidate]],
    ) -> list[tuple[str, float, Candidate]]:
        listwise_n = min(self.listwise_top_k, len(top))
        head = top[:listwise_n]
        tail = top[listwise_n:]

        body_lines: list[str] = []
        for nid, _, cand in head:
            data = graph.nodes[nid].get("data")
            if data is None:
                continue
            chain = " -> ".join(cand.causal_chain) if cand.causal_chain else "(none)"
            body_lines.append(
                f"### {data.file_path}::{data.qualname}\n"
                f"chain: {chain}\n"
                f"code:\n```python\n{(data.code or '')[:1500]}\n```"
            )
        prompt = self._listwise_tpl.format(
            issue=issue.strip()[:6000],
            candidates="\n\n".join(body_lines),
        )
        resp = self.nim.complete(
            prompt,
            mode=ReasoningMode.THINK_HIGH,
            json_schema={"type": "object"},
            temperature=0.0,
        )
        order = _parse_listwise(resp.text)
        if not order:
            return top  # fallback

        key_to_item = {}
        for nid, score, cand in head:
            data = graph.nodes[nid].get("data")
            if data is None:
                continue
            key_to_item[f"{data.file_path}::{data.qualname}"] = (nid, score, cand)

        reranked: list[tuple[str, float, Candidate]] = []
        seen: set[str] = set()
        for entry in order:
            item = key_to_item.get(entry["id"])
            if item is None:
                continue
            seen.add(entry["id"])
            nid, _old_score, cand = item
            reranked.append((nid, float(entry.get("confidence", 0.0)), cand))
        # append untouched items
        for k, item in key_to_item.items():
            if k not in seen:
                reranked.append(item)
        return reranked + tail

    @staticmethod
    def _suspect_lines(graph: nx.MultiDiGraph, cand: Candidate) -> tuple[int, int] | None:
        if not cand.suspect_statements:
            data = graph.nodes[cand.node_id].get("data") if cand.node_id in graph else None
            if data is None:
                return None
            return (data.start_line, data.end_line)
        lines = []
        for sid in cand.suspect_statements:
            if sid in graph:
                d = graph.nodes[sid]["data"]
                lines.append(d.start_line)
        if not lines:
            return None
        return (min(lines), max(lines))


def _parse_listwise(raw: str) -> list[dict]:
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return []
    ranking = obj.get("ranking") if isinstance(obj, dict) else None
    return [r for r in (ranking or []) if isinstance(r, dict) and "id" in r]


# ---------------- consistency voting ----------------


@dataclass
class VotedItem:
    function_key: str
    file_path: str
    qualname: str
    score: float
    confidence: str        # high | medium | low
    runs_appearing: int
    suspect_lines: tuple[int, int] | None
    causal_chain: list[str]


def consistency_vote(
    runs: list[list[RankedItem]],
    *,
    top_k: int = 10,
) -> list[VotedItem]:
    if not runs:
        return []
    n_runs = len(runs)

    # gather scores per function_key
    scores: dict[str, list[float]] = {}
    last_seen: dict[str, RankedItem] = {}
    counts: Counter[str] = Counter()
    for run in runs:
        seen_in_run: set[str] = set()
        for r in run[:top_k]:
            scores.setdefault(r.function_key, []).append(r.score)
            last_seen[r.function_key] = r
            if r.function_key not in seen_in_run:
                counts[r.function_key] += 1
                seen_in_run.add(r.function_key)

    voted: list[VotedItem] = []
    for key, ss in scores.items():
        k = counts[key]
        mean_score = float(np.mean(ss))
        weighted = mean_score * (k / n_runs)
        if k == n_runs:
            conf = "high"
        elif k >= max(2, n_runs - 1):
            conf = "medium"
        elif k >= 1:
            conf = "low"
        else:
            continue
        ref = last_seen[key]
        voted.append(
            VotedItem(
                function_key=key,
                file_path=ref.file_path,
                qualname=ref.qualname,
                score=weighted,
                confidence=conf,
                runs_appearing=k,
                suspect_lines=ref.suspect_lines,
                causal_chain=ref.causal_chain,
            )
        )
    voted.sort(key=lambda v: -v.score)
    return voted
