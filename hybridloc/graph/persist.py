"""Pickle-based serializer for the graph (dev-friendly; SQLite is a future opt)."""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx


def save_graph(g: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: Path) -> nx.MultiDiGraph:
    with path.open("rb") as f:
        return pickle.load(f)
