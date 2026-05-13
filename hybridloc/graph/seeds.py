"""Combine stack-frame / literal-match / concept / memory seeds into the priority queue's start set."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..pipeline.stage3_symptoms import Symptoms
from .nodes import EdgeType, NodeType, fid_concept, fid_function, fid_symptom


@dataclass
class Seed:
    node_id: str
    prior: float
    provenance: str    # "stack" | "literal" | "concept" | "memory"


# Priors from HybridLoc_v2_plan.md §5.4
PRIORS = {
    "stack": 1.0,
    "literal": 0.8,
    "concept": 0.5,
    "memory": 0.4,
    "stage1": 0.3,
}


def build_seed_set(
    g: nx.MultiDiGraph,
    symptoms: Symptoms,
    *,
    issue_embedding: np.ndarray | None = None,
    concept_top_k: int = 3,
    memory_top_k: int = 10,
    cap: int = 30,
    stage1_candidate_files: list[str] | None = None,
) -> list[Seed]:
    seeds: dict[str, Seed] = {}

    def _add(nid: str, prior: float, provenance: str) -> None:
        if nid not in g:
            return
        cur = seeds.get(nid)
        if cur is None or prior > cur.prior:
            seeds[nid] = Seed(node_id=nid, prior=prior, provenance=provenance)

    # 1) Stack-frame anchors
    for sf in symptoms.stack_frames:
        # try exact qualname first
        candidates = []
        for nid, data in g.nodes(data="data"):
            if not data or data.node_type != NodeType.FUNCTION:
                continue
            if data.file_path == sf.file and data.qualname.split(".")[-1] == sf.func:
                if data.start_line <= sf.line <= data.end_line:
                    candidates.append(nid)
        for nid in candidates:
            _add(nid, PRIORS["stack"], "stack")

    # 2) Literal matches (exception types, named APIs, behaviors): substring match
    keys = symptoms.keywords()
    keys_lower = [k.lower() for k in keys if k]
    if keys_lower:
        for nid, data in g.nodes(data="data"):
            if not data or data.node_type != NodeType.FUNCTION:
                continue
            hay = (data.qualname + " " + data.docstring).lower()
            for k in keys_lower:
                if k and k in hay:
                    _add(nid, PRIORS["literal"], "literal")
                    break

    # 3) Concept seeds — pick top-k clusters by centroid cosine to the issue embedding
    if issue_embedding is not None:
        concept_scores: list[tuple[str, float]] = []
        for nid, data in g.nodes(data="data"):
            if not data or data.node_type != NodeType.CONCEPT:
                continue
            centroid = data.extra.get("centroid")
            if centroid is None:
                continue
            c = np.asarray(centroid, dtype=np.float32)
            sim = float(np.dot(c, issue_embedding))
            concept_scores.append((nid, sim))
        concept_scores.sort(key=lambda x: -x[1])
        for cid, _ in concept_scores[:concept_top_k]:
            # add all member functions of those clusters
            for _u, v, _k in g.in_edges(cid, keys=True):
                if v == cid:
                    continue
                if g.nodes[v].get("data") and g.nodes[v]["data"].node_type == NodeType.FUNCTION:
                    _add(v, PRIORS["concept"], "concept")

    # 4) Stage 1 fallback seeds — always add all functions from the top-K candidate files
    # This guarantees a non-empty seed set even when symptoms are weak and concepts are skipped.
    if stage1_candidate_files:
        for nid, data in g.nodes(data="data"):
            if not data or data.node_type != NodeType.FUNCTION:
                continue
            if data.file_path in stage1_candidate_files:
                _add(nid, PRIORS.get("stage1", 0.3), "stage1")

    # cap by prior
    ordered = sorted(seeds.values(), key=lambda s: -s.prior)
    return ordered[:cap]


def attach_symptom_nodes(g: nx.MultiDiGraph, symptoms: Symptoms) -> list[str]:
    """Add Symptom nodes + SYMPTOM_OF edges (function->symptom). Returns symptom node ids."""
    sym_ids: list[str] = []
    seen: set[str] = set()
    for label_set, kind in (
        (symptoms.exception_types, "exception"),
        (symptoms.error_messages, "error_msg"),
        (symptoms.behaviors, "behavior"),
        (symptoms.api_calls_named, "api"),
    ):
        for label in label_set:
            if not label or label in seen:
                continue
            seen.add(label)
            sid = fid_symptom(f"{kind}::{label}")
            g.add_node(sid, data=_symptom_node(label, kind))
            sym_ids.append(sid)
    return sym_ids


def _symptom_node(label: str, kind: str):
    from .nodes import NodeData

    return NodeData(
        node_type=NodeType.SYMPTOM,
        name=label,
        extra={"kind": kind},
    )
