"""Determinism + ordering of the Stage-4 priority queue on a tiny graph."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import numpy as np

from hybridloc.graph.nodes import EdgeType, NodeData, NodeType, fid_function
from hybridloc.graph.seeds import Seed
from hybridloc.pipeline.causal import CausalUpdate
from hybridloc.pipeline.stage3_symptoms import Symptoms
from hybridloc.pipeline.stage4_traversal import TraversalConfig, Traverser


def _tiny_graph() -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = nx.MultiDiGraph()
    for q in ["A", "B", "C"]:
        nid = fid_function("m.py", q)
        g.add_node(
            nid,
            data=NodeData(
                node_type=NodeType.FUNCTION,
                name=q,
                file_path="m.py",
                qualname=q,
                start_line=1,
                end_line=10,
                code=f"def {q}(): pass",
                docstring=f"function {q}",
            ),
        )
    g.add_edge(
        fid_function("m.py", "A"),
        fid_function("m.py", "B"),
        key=EdgeType.INVOKE.value,
        edge_type=EdgeType.INVOKE,
    )
    return g


class _StubDense:
    def encode(self, texts: list[str]) -> np.ndarray:
        # any text containing 'A' embeds high cosine with the issue (which we set to 'A')
        return np.asarray(
            [[1.0, 0.0] if "A" in t else [0.0, 1.0] for t in texts],
            dtype=np.float32,
        )


def test_traverser_picks_seed_first_then_callee(tmp_path: Path):
    g = _tiny_graph()
    reasoner = MagicMock()
    reasoner.update_chains.return_value = CausalUpdate()
    cfg = TraversalConfig(max_iterations=10, max_think_high_calls=0, causal_every_k_iters=1)

    traverser = Traverser(
        g,
        repo_root=tmp_path,
        reasoner=reasoner,
        dense=_StubDense(),
        config=cfg,
    )
    seeds = [Seed(node_id=fid_function("m.py", "A"), prior=1.0, provenance="stack")]
    out = traverser.run(issue="A", symptoms=Symptoms(), seeds=seeds)
    out_qns = [g.nodes[c.node_id]["data"].qualname for c in out]
    # 'A' is the seed and has the highest semantic score, so it must be first
    assert out_qns[0] == "A"
    # 'B' should also surface since A INVOKEs B
    assert "B" in out_qns


def test_termination_reasons(tmp_path: Path):
    g = _tiny_graph()
    reasoner = MagicMock()
    reasoner.update_chains.return_value = CausalUpdate()
    cfg = TraversalConfig(max_iterations=2, max_think_high_calls=0, causal_every_k_iters=1)
    traverser = Traverser(g, repo_root=tmp_path, reasoner=reasoner, dense=_StubDense(), config=cfg)
    seeds = [Seed(node_id=fid_function("m.py", "A"), prior=1.0, provenance="stack")]
    traverser.run(issue="A", symptoms=Symptoms(), seeds=seeds)
    assert traverser.termination_reason in {"iter_cap", "queue_empty", "stable"}
