"""Combine stack-frame / literal-match / concept / memory seeds into the priority queue's start set."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..pipeline.stage3_symptoms import Symptoms
from .nodes import EdgeType, NodeType, fid_concept, fid_function, fid_symptom


@dataclass
class Seed:
    node_id: str
    prior: float
    provenance: str    # "stack" | "literal" | "concept" | "memory" | "stage1"


# Priors from HybridLoc_v2_plan.md §5.4
PRIORS = {
    "stack": 1.0,
    "literal": 0.8,
    "concept": 0.5,
    "memory": 0.4,
    "stage1": 0.3,
}

# LocAgent-style soft penalty for test files. They still seed if LLM ranked
# them highly, but cannot dominate when implementation files are available.
_TEST_FILE_PENALTY = 0.5
_TEST_FILE_RE = re.compile(
    r"(^|/)tests?/"          # any /tests/ or /test/ directory
    r"|(/|^)tests?\.py$"     # files literally named tests.py or test.py
    r"|(/|^)test_[^/]*\.py$" # test_*.py
    r"|_test\.py$"           # *_test.py
)


def _is_test_file(path: str) -> bool:
    return bool(_TEST_FILE_RE.search(path or ""))


def _seeds_per_file_budget(rank: int) -> int:
    """Rank-decayed allocation: top files get more seeds, but no monopoly.
    Rank 0→8, 1→5, 2→4, 3→3, 4+→2. Caps any single file's contribution."""
    return max(2, math.ceil(8 * math.exp(-0.4 * rank)))


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

    # 4) Stage 1 fallback seeds — rank-weighted prior with PER-FILE budget.
    # Prevents one file (e.g. a test file with 30 methods) from monopolizing the seed cap.
    # Test files get a soft penalty so they don't dominate implementation files.
    if stage1_candidate_files:
        n_files = len(stage1_candidate_files)
        rank_map = {f: i for i, f in enumerate(stage1_candidate_files)}

        # group function nodes by file_path
        funcs_by_file: dict[str, list[tuple[str, "any"]]] = {}
        for nid, data in g.nodes(data="data"):
            if not data or data.node_type != NodeType.FUNCTION:
                continue
            if data.file_path in rank_map:
                funcs_by_file.setdefault(data.file_path, []).append((nid, data))

        # for each file, score each function vs issue (if embedding available) so
        # we pick the file's MOST relevant N functions, not arbitrary ones
        for file_path, funcs in funcs_by_file.items():
            rank = rank_map[file_path]
            budget = _seeds_per_file_budget(rank)
            base_prior = 0.35 - 0.20 * (rank / max(1, n_files - 1))
            if _is_test_file(file_path):
                base_prior *= _TEST_FILE_PENALTY

            # rank this file's functions: by issue embedding similarity if available,
            # else fall back to declaration order
            if issue_embedding is not None:
                def _sim(item):
                    _, d = item
                    centroid = (d.extra or {}).get("centroid")
                    if centroid is None:
                        return 0.0
                    c = np.asarray(centroid, dtype=np.float32)
                    return float(np.dot(c, issue_embedding))
                ranked_funcs = sorted(funcs, key=_sim, reverse=True)
            else:
                ranked_funcs = funcs

            for nid, _ in ranked_funcs[:budget]:
                _add(nid, base_prior, "stage1")

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
