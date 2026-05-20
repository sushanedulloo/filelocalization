"""Stage 4: priority-guided graph traversal with abductive causal reasoning.

This is the heart of HybridLoc. The traversal:
1. Seeds the queue with `Seed`s (Stage 3 output)
2. Pops the highest-priority action, executes it (read code, expand neighbors,
   drill into statements)
3. Every K iterations, asks the LLM to construct causal chains for newly-read
   functions (Think-High, ~12 calls per issue budgeted)
4. Terminates on stability, iteration cap, or LLM call cap.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from ..graph.dataflow import find_function_node, materialize_dataflow_for_function
from ..graph.nodes import EdgeType, NodeType
from ..graph.seeds import Seed
from ..pipeline.actions import Action, ActionKind
from ..pipeline.causal import AbductiveReasoner, CausalUpdate
from ..pipeline.stage3_symptoms import Symptoms
from ..retrieval.dense import DenseRetriever


@dataclass
class TraversalConfig:
    w_sem: float = 1.0
    w_chain: float = 0.8
    w_seed: float = 0.6
    w_recent: float = 0.2
    w_depth: float = 0.4
    w_visited: float = 2.0
    chain_decay_iterations: int = 10
    drill_chain_score: float = 7.0
    max_iterations: int = 30
    max_think_high_calls: int = 12
    queue_cap: int = 1000
    causal_every_k_iters: int = 3
    candidates_per_causal_call: int = 3
    output_top_n: int = 15


@dataclass
class Candidate:
    node_id: str
    score: float = 0.0
    distance: int = 0
    causal_chain: list[str] = field(default_factory=list)
    chain_score: float = 0.0
    chain_iter: int = -1
    suspect_statements: list[str] = field(default_factory=list)


class Traverser:
    def __init__(
        self,
        graph: nx.MultiDiGraph,
        *,
        repo_root: Path,
        reasoner: AbductiveReasoner,
        dense: DenseRetriever | None,
        config: TraversalConfig | None = None,
    ):
        self.g = graph
        self.repo_root = repo_root
        self.reasoner = reasoner
        self.dense = dense
        self.cfg = config or TraversalConfig()

        self._heap: list[tuple[float, int, Action]] = []
        self._counter = 0           # tiebreaker
        self._visited: set[str] = set()
        self._distances: dict[str, int] = {}
        self._candidates: dict[str, Candidate] = {}
        self._issue_emb: np.ndarray | None = None
        self._issue_text: str = ""
        self._symptoms_text: str = ""
        self._iter: int = 0
        self._think_calls: int = 0
        self._termination: str = ""
        self._buffered_for_causal: list[tuple[str, str]] = []

    # ------------- public API -------------

    def run(
        self,
        *,
        issue: str,
        symptoms: Symptoms,
        seeds: list[Seed],
    ) -> list[Candidate]:
        self._issue_text = issue
        self._symptoms_text = _fmt_symptoms(symptoms)
        if self.dense is not None:
            self._issue_emb = self.dense.encode([issue])[0]

        from ..log import info
        info(f"[Traversal] Starting with {len(seeds)} seeds")
        for s in seeds:
            data = self.g.nodes[s.node_id].get("data") if s.node_id in self.g else None
            name = data.qualname if data else s.node_id
            info(f"[Traversal] Seed: {name} (prior={s.prior:.2f}, source={s.provenance})")
            self._distances[s.node_id] = 0
            self._push(
                Action(
                    kind=ActionKind.READ_FUNCTION,
                    target_id=s.node_id,
                    depth=0,
                    seed_prior=s.prior,
                )
            )

        stable_iters = 0
        last_top3: tuple[str, ...] = ()

        from ..log import info
        while self._heap and self._iter < self.cfg.max_iterations:
            self._iter += 1
            _, priority_neg, action = heapq.heappop(self._heap)
            if action.target_id in self._visited and action.kind == ActionKind.READ_FUNCTION:
                continue
            data = self.g.nodes[action.target_id].get("data") if action.target_id in self.g else None
            name = data.qualname if data else action.target_id
            info(f"[Traversal] iter={self._iter:02d} action={action.kind.value} target={name} depth={action.depth} queue_size={len(self._heap)}")
            self._execute(action)

            # periodically run causal reasoning
            if (
                self._iter % self.cfg.causal_every_k_iters == 0
                and self._buffered_for_causal
                and self._think_calls < self.cfg.max_think_high_calls
            ):
                self._run_causal()

            top3 = tuple(c.node_id for c in self._top(3))
            if top3 == last_top3:
                stable_iters += 1
            else:
                stable_iters = 0
                last_top3 = top3
            if stable_iters >= 3 and self._iter >= 6:
                self._termination = "stable"
                break

        if not self._termination:
            if self._iter >= self.cfg.max_iterations:
                self._termination = "iter_cap"
            elif not self._heap:
                self._termination = "queue_empty"

        # final flush of pending causal calls
        if self._buffered_for_causal and self._think_calls < self.cfg.max_think_high_calls:
            self._run_causal()

        from ..log import info
        top = self._top(self.cfg.output_top_n)
        info(f"[Traversal] Done — termination={self._termination}, iters={self._iter}, think_calls={self._think_calls}, candidates={len(top)}")
        for i, c in enumerate(top[:5]):
            data = self.g.nodes[c.node_id].get("data") if c.node_id in self.g else None
            name = f"{data.file_path}::{data.qualname}" if data else c.node_id
            info(f"[Traversal] Top-{i+1}: {name} score={c.score:.3f} chain_score={c.chain_score:.1f}")
        return top

    @property
    def termination_reason(self) -> str:
        return self._termination

    @property
    def think_calls(self) -> int:
        return self._think_calls

    # ------------- internals -------------

    def _push(self, action: Action) -> None:
        if len(self._heap) >= self.cfg.queue_cap:
            return
        action.iteration_added = self._iter
        priority = self._compute_priority(action)
        self._counter += 1
        heapq.heappush(self._heap, (-priority, self._counter, action))

    def _compute_priority(self, action: Action) -> float:
        nid = action.target_id
        data = self.g.nodes[nid].get("data") if nid in self.g else None
        if data is None:
            return -1.0
        # semantic
        sem = self._semantic_score(nid, data)
        # chain — with distal-cause boost (GraphLocator §4.3): functions
        # deeper in the causal chain (further from the symptom) get
        # multiplied. A function at chain depth 3 is more often the
        # actual edit site than one at chain depth 1, which usually
        # describes the symptom rather than the root cause.
        cand = self._candidates.get(nid)
        chain_score = 0.0
        if cand and cand.chain_iter >= 0:
            age = self._iter - cand.chain_iter
            decay = max(0.0, 1.0 - age / max(1, self.cfg.chain_decay_iterations))
            base = (cand.chain_score / 10.0) * decay
            # distal-cause multiplier (GraphLocator §4.3)
            chain_len = len(cand.causal_chain) if cand.causal_chain else 1
            distal_boost = 1.0 + 0.30 * math.log(max(1, chain_len))
            chain_score = base * distal_boost
        # recency: just-added actions decay slightly
        recency = max(0.0, 1.0 - (self._iter - action.iteration_added) / 5.0)
        # depth penalty
        depth_pen = max(0, action.depth - 3)
        visited_pen = 1.0 if nid in self._visited else 0.0

        return (
            self.cfg.w_sem * sem
            + self.cfg.w_chain * chain_score
            + self.cfg.w_seed * action.seed_prior
            + self.cfg.w_recent * recency
            - self.cfg.w_depth * depth_pen
            - self.cfg.w_visited * visited_pen
            + action.boost
        )

    def _semantic_score(self, nid: str, data: Any) -> float:
        if self._issue_emb is None or self.dense is None:
            return 0.0
        text = f"{data.qualname or data.name}\n{data.docstring or ''}\n{data.code or ''}"[:4000]
        try:
            emb = self.dense.encode([text])[0]
        except Exception:
            return 0.0
        return float(np.dot(emb, self._issue_emb))

    def _execute(self, action: Action) -> None:
        nid = action.target_id
        data = self.g.nodes[nid].get("data") if nid in self.g else None
        if data is None:
            return

        if action.kind == ActionKind.READ_FUNCTION:
            if data.node_type != NodeType.FUNCTION:
                # action decomposition: for class/file targets, score and push top methods
                self._action_decompose(nid, data, action)
                return
            self._read_function(nid, data, action)

        elif action.kind == ActionKind.EXPAND_CALL:
            for u, v, k in self.g.out_edges(nid, keys=True):
                if k == EdgeType.INVOKE.value:
                    self._enqueue_neighbor(v, action)

        elif action.kind == ActionKind.EXPAND_INHERIT:
            for u, v, k in self.g.out_edges(nid, keys=True):
                if k == EdgeType.INHERIT.value:
                    self._enqueue_neighbor(v, action)
            for u, v, k in self.g.in_edges(nid, keys=True):
                if k == EdgeType.INHERIT.value:
                    self._enqueue_neighbor(u, action)

        elif action.kind == ActionKind.EXPAND_CO_EVOLVED:
            file_id = self._function_file_id(nid)
            if file_id:
                for u, v, k in self.g.out_edges(file_id, keys=True):
                    if k == EdgeType.CO_EVOLVED.value:
                        # push READ_FUNCTION for each function in that file
                        for _, fn, fk in self.g.out_edges(v, keys=True):
                            if (
                                fk == EdgeType.CONTAIN.value
                                and self.g.nodes[fn]["data"].node_type
                                == NodeType.FUNCTION
                            ):
                                self._enqueue_neighbor(fn, action)

        elif action.kind == ActionKind.DRILL_STATEMENTS:
            self._drill_statements(nid, data)

        elif action.kind == ActionKind.EXPAND_DU:
            for u, v, k in self.g.out_edges(nid, keys=True):
                if k == EdgeType.DEF_USE.value:
                    self._enqueue_neighbor(v, action)

    def _read_function(self, nid: str, data: Any, action: Action) -> None:
        self._visited.add(nid)
        # load code if not already there
        if not data.code:
            try:
                src = (self.repo_root / data.file_path).read_text(errors="replace")
                lines = src.splitlines()
                code = "\n".join(lines[max(0, data.start_line - 1) : data.end_line])
                data.code = code
            except OSError:
                data.code = ""

        cand = self._candidates.setdefault(nid, Candidate(node_id=nid))
        cand.distance = action.depth
        # auto-expand call/inherit neighbors at moderate priority
        self._push(
            Action(
                kind=ActionKind.EXPAND_CALL,
                target_id=nid,
                depth=action.depth,
                seed_prior=action.seed_prior * 0.9,
            )
        )
        self._push(
            Action(
                kind=ActionKind.EXPAND_INHERIT,
                target_id=nid,
                depth=action.depth,
                seed_prior=action.seed_prior * 0.7,
            )
        )
        # buffer for causal chain reasoning
        key = f"{data.file_path}::{data.qualname}"
        self._buffered_for_causal.append((key, data.code or ""))

    def _action_decompose(self, nid: str, data: Any, action: Action) -> None:
        """For class/file: score child functions by skeleton, push top-3 as READ_FUNCTION."""
        children: list[tuple[str, float]] = []
        for u, v, k in self.g.out_edges(nid, keys=True):
            if k != EdgeType.CONTAIN.value:
                continue
            child = self.g.nodes[v].get("data")
            if child is None or child.node_type != NodeType.FUNCTION:
                continue
            sem = self._semantic_score(v, child)
            children.append((v, sem))
        children.sort(key=lambda x: -x[1])
        for child_id, _ in children[:3]:
            self._enqueue_neighbor(child_id, action)

    def _enqueue_neighbor(self, nid: str, parent: Action, *, boost: float = 0.0) -> None:
        new_depth = parent.depth + 1
        if nid in self._distances:
            new_depth = min(new_depth, self._distances[nid] + 1)
        self._distances[nid] = min(self._distances.get(nid, new_depth), new_depth)
        self._push(
            Action(
                kind=ActionKind.READ_FUNCTION,
                target_id=nid,
                depth=new_depth,
                seed_prior=parent.seed_prior * 0.85,
                boost=boost,
            )
        )

    def _function_file_id(self, fn_id: str) -> str | None:
        for u, v, k in self.g.in_edges(fn_id, keys=True):
            if k == EdgeType.CONTAIN.value:
                pdata = self.g.nodes[u].get("data")
                if pdata and pdata.node_type == NodeType.FILE:
                    return u
                if pdata and pdata.node_type == NodeType.CLASS:
                    # one more level up
                    for u2, _, k2 in self.g.in_edges(u, keys=True):
                        if k2 == EdgeType.CONTAIN.value:
                            pd2 = self.g.nodes[u2].get("data")
                            if pd2 and pd2.node_type == NodeType.FILE:
                                return u2
        return None

    def _drill_statements(self, fn_id: str, data: Any) -> None:
        try:
            src = (self.repo_root / data.file_path).read_bytes()
        except OSError:
            return
        node = find_function_node(src, data.qualname)
        if node is None:
            return
        n_added = materialize_dataflow_for_function(
            self.g, src, data.file_path, data.qualname, node
        )
        cand = self._candidates.setdefault(fn_id, Candidate(node_id=fn_id))
        # collect the top suspect statements (those with highest cosine to issue)
        if n_added and self._issue_emb is not None and self.dense is not None:
            stmt_ids = [
                v
                for u, v, k in self.g.out_edges(fn_id, keys=True)
                if k == EdgeType.CONTAIN.value
                and self.g.nodes[v]["data"].node_type == NodeType.STATEMENT
            ]
            if stmt_ids:
                texts = [self.g.nodes[s]["data"].code for s in stmt_ids]
                embs = self.dense.encode(texts)
                sims = embs @ self._issue_emb
                order = np.argsort(-sims)
                cand.suspect_statements = [stmt_ids[i] for i in order[:5]]

    def _run_causal(self) -> None:
        if not self._buffered_for_causal:
            return
        from ..log import info
        batch = self._buffered_for_causal[: self.cfg.candidates_per_causal_call * 3]
        self._buffered_for_causal = self._buffered_for_causal[len(batch) :]
        info(f"[Stage 4] Causal reasoning call #{self._think_calls+1} on {len(batch)} functions (iter={self._iter}) ...")
        cig = self._mermaid_for_top(8)
        update: CausalUpdate = self.reasoner.update_chains(
            issue=self._issue_text,
            symptoms_text=self._symptoms_text,
            cig_mermaid=cig,
            candidates=batch,
        )
        self._think_calls += 1

        # apply chains
        for ch in update.chains:
            nid = self._function_key_to_id(ch.function_key)
            if nid is None:
                continue
            cand = self._candidates.setdefault(nid, Candidate(node_id=nid))
            cand.causal_chain = ch.chain
            cand.chain_score = ch.score
            cand.chain_iter = self._iter
            # if chain is strong, drill statements
            if ch.score >= self.cfg.drill_chain_score:
                self._push(
                    Action(
                        kind=ActionKind.DRILL_STATEMENTS,
                        target_id=nid,
                        depth=cand.distance,
                        seed_prior=0.5,
                    )
                )

        # boosted exploration suggestions — lenient key resolution because
        # the causal LLM often produces names with module prefixes or class
        # names that don't exactly match our qualname format.
        for k in update.next_to_explore:
            nids = self._resolve_function_key(k)
            for nid in nids:
                self._push(
                    Action(
                        kind=ActionKind.READ_FUNCTION,
                        target_id=nid,
                        depth=self._distances.get(nid, 2),
                        seed_prior=0.55,   # raised from 0.4 — must outrank stage1 seeds (max 0.35)
                        boost=0.6,         # raised from 0.5 — gives 1.15 effective prior
                    )
                )

    def _function_key_to_id(self, key: str) -> str | None:
        if "::" not in key:
            return None
        path, qual = key.split("::", 1)
        from ..graph.nodes import fid_function

        nid = fid_function(path, qual)
        return nid if nid in self.g else None

    def _resolve_function_key(self, key: str) -> list[str]:
        """Lenient resolution of an LLM-suggested key like 'path::qual'.

        Tries multiple variants because the LLM often produces keys that
        don't exactly match our qualnames:
          - "django/core/validators.py::URLValidator"  → matches __call__,
            __init__, and any method of URLValidator
          - "core/validators.py::URLValidator.validate" (missing django/)
            → resolves to django/core/validators.py::URLValidator.validate
          - "path::Class.method" where only Class exists → matches all
            methods of that class

        Returns a list of node ids (may be empty if no match).
        """
        from ..graph.nodes import NodeType, fid_function
        if "::" not in key:
            return []
        path_raw, qual_raw = key.split("::", 1)
        path = path_raw.lstrip("/")
        qual = qual_raw.strip()

        # Exact match
        nid = fid_function(path, qual)
        if nid in self.g:
            return [nid]

        results: list[str] = []
        # Match by qualname suffix + path suffix — handles missing/extra prefix
        for nid_iter, data in self.g.nodes(data="data"):
            if not data or data.node_type != NodeType.FUNCTION:
                continue
            if not data.file_path or not data.qualname:
                continue
            # Path is a suffix match (e.g. "core/validators.py" matches
            # "django/core/validators.py")
            if not (data.file_path == path
                    or data.file_path.endswith("/" + path)
                    or path.endswith("/" + data.file_path)):
                continue
            # Qualname: exact, or "Class.method" of suggested class, or
            # method match in suggested class
            if data.qualname == qual:
                results.append(nid_iter); continue
            # If the LLM suggested just the class name, accept all methods
            if "." not in qual and data.qualname.startswith(qual + "."):
                results.append(nid_iter); continue
            # If the LLM suggested "Class.method" but graph has it dotted
            # under a different parent (rare), match the last two segments
            qual_tail = ".".join(qual.split(".")[-2:])
            data_tail = ".".join(data.qualname.split(".")[-2:])
            if qual_tail == data_tail:
                results.append(nid_iter); continue
        return results[:5]   # cap to avoid flooding the queue

    def _mermaid_for_top(self, n: int) -> str:
        cands = self._top(n)
        if not cands:
            return "graph TD\n  (empty)"
        lines = ["graph TD"]
        for c in cands:
            data = self.g.nodes[c.node_id].get("data")
            if data is None:
                continue
            label = f"{Path(data.file_path).name}::{data.qualname}"
            lines.append(f"  {self._mermaid_id(c.node_id)}[\"{label}\"]")
        return "\n".join(lines)

    @staticmethod
    def _mermaid_id(s: str) -> str:
        return "n" + str(abs(hash(s)) % 10**8)

    def _top(self, n: int) -> list[Candidate]:
        scored: list[Candidate] = []
        for nid, cand in self._candidates.items():
            data = self.g.nodes[nid].get("data") if nid in self.g else None
            if data is None or data.node_type != NodeType.FUNCTION:
                continue
            sem = self._semantic_score(nid, data)
            cand.score = (
                self.cfg.w_sem * sem
                + self.cfg.w_chain * (cand.chain_score / 10.0)
                + self.cfg.w_seed * 0.0
                - self.cfg.w_depth * max(0, cand.distance - 3)
            )
            scored.append(cand)
        scored.sort(key=lambda c: -c.score)
        return scored[:n]


def _fmt_symptoms(s: Symptoms) -> str:
    parts: list[str] = []
    if s.exception_types:
        parts.append("Exceptions: " + ", ".join(s.exception_types))
    if s.error_messages:
        parts.append("Errors: " + " | ".join(s.error_messages[:5]))
    if s.stack_frames:
        parts.append(
            "Stack: "
            + "; ".join(f"{sf.file}:{sf.line} in {sf.func}" for sf in s.stack_frames)
        )
    if s.behaviors:
        parts.append("Behaviors: " + ", ".join(s.behaviors))
    if s.api_calls_named:
        parts.append("APIs: " + ", ".join(s.api_calls_named))
    return "\n".join(parts)
