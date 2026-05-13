"""RepoLens: per-function NL summary -> embedding -> K-means -> labeled clusters."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..llm.nim_client import NIMClient, ReasoningMode
from ..parsing.skeleton import Skeleton
from ..retrieval.dense import DenseRetriever
from .nodes import EdgeType, NodeData, NodeType, fid_concept, fid_function

_SUMMARY_PROMPT = (
    "Summarize what this Python function does in ONE short sentence "
    "(<= 20 words). Return only the sentence.\n\n"
    "FILE: {file}\n"
    "FUNCTION: {qualname}\n"
    "SIGNATURE: {signature}\n"
    "DOCSTRING: {docstring}\n"
)

_LABEL_PROMPT = (
    "Below are short summaries of {n} related functions in a codebase. "
    "Give a 2-5 word LABEL for the shared concept they implement. "
    "Output only the label.\n\n"
    "{summaries}\n"
)


@dataclass
class Concept:
    label: str
    member_function_ids: list[str] = field(default_factory=list)
    centroid: np.ndarray | None = None


async def summarize_functions(
    skels: list[Skeleton], nim: NIMClient
) -> dict[str, str]:
    """Returns {function_qualname_with_path: one_sentence_summary}."""
    prompts: list[str] = []
    keys: list[str] = []
    for sk in skels:
        for f in sk.functions:
            keys.append(f"{sk.file_path}::{f.qualname}")
            prompts.append(
                _SUMMARY_PROMPT.format(
                    file=sk.file_path,
                    qualname=f.qualname,
                    signature=f.signature,
                    docstring=f.docstring,
                )
            )
        for c in sk.classes:
            for m in c.methods:
                keys.append(f"{sk.file_path}::{m.qualname}")
                prompts.append(
                    _SUMMARY_PROMPT.format(
                        file=sk.file_path,
                        qualname=m.qualname,
                        signature=m.signature,
                        docstring=m.docstring,
                    )
                )
    responses = await nim.acomplete_many(prompts, mode=ReasoningMode.NON_THINK, desc="concept extraction")
    return {k: r.text.strip() for k, r in zip(keys, responses)}


def cluster_summaries(
    summaries: dict[str, str],
    *,
    dense: DenseRetriever,
    min_k: int = 5,
    max_k: int = 500,
) -> list[Concept]:
    if not summaries:
        return []
    keys = list(summaries.keys())
    texts = [summaries[k] for k in keys]
    emb = dense.encode(texts)

    n = len(keys)
    k = max(min_k, min(max_k, int(math.sqrt(n))))
    centroids, labels = _kmeans(emb, k=k, n_iter=20)

    concepts: list[Concept] = []
    for cid in range(k):
        member_idxs = np.where(labels == cid)[0]
        if len(member_idxs) == 0:
            continue
        concepts.append(
            Concept(
                label=f"cluster_{cid:03d}",
                member_function_ids=[keys[i] for i in member_idxs.tolist()],
                centroid=centroids[cid],
            )
        )
    return concepts


async def label_concepts(
    concepts: list[Concept], summaries: dict[str, str], nim: NIMClient,
    *, max_examples: int = 8,
) -> None:
    prompts: list[str] = []
    for c in concepts:
        examples = c.member_function_ids[:max_examples]
        body = "\n".join(f"- {summaries.get(k, '')}" for k in examples)
        prompts.append(_LABEL_PROMPT.format(n=len(examples), summaries=body))
    responses = await nim.acomplete_many(prompts, mode=ReasoningMode.NON_THINK, desc="cluster labeling")
    for c, r in zip(concepts, responses):
        c.label = r.text.strip().splitlines()[0][:60] or c.label


def add_concept_nodes(g, concepts: list[Concept], summaries: dict[str, str]) -> None:
    for c in concepts:
        cid = fid_concept(c.label)
        g.add_node(
            cid,
            data=NodeData(
                node_type=NodeType.CONCEPT,
                name=c.label,
                extra={
                    "centroid": c.centroid.tolist() if c.centroid is not None else None,
                    "n_members": len(c.member_function_ids),
                },
            ),
        )
        for fk in c.member_function_ids:
            if "::" not in fk:
                continue
            file_path, qualname = fk.split("::", 1)
            fn_id = fid_function(file_path, qualname)
            if fn_id in g:
                g.add_edge(
                    fn_id, cid, key=EdgeType.CONCEPT_OF.value, edge_type=EdgeType.CONCEPT_OF
                )


# ---------- tiny kmeans (no sklearn dep just for this) ----------


def _kmeans(x: np.ndarray, *, k: int, n_iter: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if k >= n:
        k = max(1, n)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # cosine similarity since vectors are unit-normed
        sims = x @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if not mask.any():
                # re-seed empty cluster
                centroids[c] = x[rng.integers(0, n)]
            else:
                v = x[mask].mean(axis=0)
                norm = np.linalg.norm(v) + 1e-9
                centroids[c] = v / norm
    return centroids, labels
