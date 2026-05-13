"""Traversal actions. Each carries the data needed to compute its priority."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    READ_FUNCTION = "read_function"
    EXPAND_CALL = "expand_call"
    EXPAND_INHERIT = "expand_inherit"
    EXPAND_CO_EVOLVED = "expand_co_evolved"
    DRILL_STATEMENTS = "drill_statements"
    EXPAND_DU = "expand_du"


@dataclass
class Action:
    kind: ActionKind
    target_id: str          # node id in the graph
    depth: int              # hops from any seed
    seed_prior: float       # max prior of any seed this action descends from
    boost: float = 0.0      # extra additive priority (e.g. from causal next_to_explore)
    iteration_added: int = 0
