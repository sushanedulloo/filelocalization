"""Heterogeneous graph builder. LocAgent-style structural edges."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from ..parsing.skeleton import Skeleton
from .callgraph import build_call_edges
from .nodes import (
    EdgeType,
    NodeData,
    NodeType,
    fid_class,
    fid_dir,
    fid_file,
    fid_function,
)


def build_structural_graph(
    skels: list[Skeleton],
    repo_root: Path | None = None,
) -> nx.MultiDiGraph:
    """Builds Directory→File→Class→Function with CONTAIN/IMPORT/INVOKE/INHERIT."""
    g: nx.MultiDiGraph = nx.MultiDiGraph()

    # nodes: directories (from path components), files
    dir_set: set[str] = set()
    file_to_qualnames: dict[str, set[str]] = {}
    file_to_source: dict[str, bytes] = {}

    for sk in skels:
        parts = sk.file_path.split("/")
        for i in range(1, len(parts)):
            dir_set.add("/".join(parts[:i]))
        # add file node
        g.add_node(
            fid_file(sk.file_path),
            data=NodeData(
                node_type=NodeType.FILE,
                name=parts[-1],
                file_path=sk.file_path,
            ),
        )
        # contain edge from parent dir
        parent_dir = "/".join(parts[:-1]) if len(parts) > 1 else ""
        if parent_dir:
            g.add_edge(
                fid_dir(parent_dir),
                fid_file(sk.file_path),
                key=EdgeType.CONTAIN.value,
                edge_type=EdgeType.CONTAIN,
            )

        qualnames: set[str] = set()
        # classes & methods
        for c in sk.classes:
            cls_id = fid_class(sk.file_path, c.name)
            g.add_node(
                cls_id,
                data=NodeData(
                    node_type=NodeType.CLASS,
                    name=c.name,
                    file_path=sk.file_path,
                    qualname=c.name,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    docstring=c.docstring,
                ),
            )
            g.add_edge(
                fid_file(sk.file_path),
                cls_id,
                key=EdgeType.CONTAIN.value,
                edge_type=EdgeType.CONTAIN,
            )
            for base in c.bases:
                # we'll resolve INHERIT at the end (need name->classid map)
                g.nodes[cls_id]["data"].extra.setdefault("bases", []).append(base)
            for m in c.methods:
                fn_id = fid_function(sk.file_path, m.qualname)
                g.add_node(
                    fn_id,
                    data=NodeData(
                        node_type=NodeType.FUNCTION,
                        name=m.name,
                        file_path=sk.file_path,
                        qualname=m.qualname,
                        start_line=m.start_line,
                        end_line=m.end_line,
                        docstring=m.docstring,
                        extra={"signature": m.signature},
                    ),
                )
                g.add_edge(
                    cls_id,
                    fn_id,
                    key=EdgeType.CONTAIN.value,
                    edge_type=EdgeType.CONTAIN,
                )
                qualnames.add(m.qualname)

        # module-level functions
        for f in sk.functions:
            fn_id = fid_function(sk.file_path, f.qualname)
            g.add_node(
                fn_id,
                data=NodeData(
                    node_type=NodeType.FUNCTION,
                    name=f.name,
                    file_path=sk.file_path,
                    qualname=f.qualname,
                    start_line=f.start_line,
                    end_line=f.end_line,
                    docstring=f.docstring,
                    extra={"signature": f.signature},
                ),
            )
            g.add_edge(
                fid_file(sk.file_path),
                fn_id,
                key=EdgeType.CONTAIN.value,
                edge_type=EdgeType.CONTAIN,
            )
            qualnames.add(f.qualname)

        file_to_qualnames[sk.file_path] = qualnames

        # IMPORT edges (best-effort: parse "from X import Y" / "import X")
        for imp in sk.imports:
            target = _import_target(imp)
            if target:
                g.add_edge(
                    fid_file(sk.file_path),
                    fid_file(target),  # may dangle; we'll prune below
                    key=EdgeType.IMPORT.value,
                    edge_type=EdgeType.IMPORT,
                )

    # add directory nodes
    for d in dir_set:
        g.add_node(
            fid_dir(d),
            data=NodeData(node_type=NodeType.DIRECTORY, name=d.split("/")[-1], file_path=d),
        )

    # parent->child dir edges
    for d in dir_set:
        parts = d.split("/")
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            if parent in dir_set:
                g.add_edge(
                    fid_dir(parent),
                    fid_dir(d),
                    key=EdgeType.CONTAIN.value,
                    edge_type=EdgeType.CONTAIN,
                )

    # prune dangling IMPORT edges (target file not in repo)
    file_ids = {fid_file(p) for p in file_to_qualnames}
    drop = [
        (u, v, k)
        for u, v, k in g.edges(keys=True)
        if k == EdgeType.IMPORT.value and v not in file_ids
    ]
    g.remove_edges_from(drop)

    # INHERIT edges (best-effort: match class short-name to a class in this repo)
    short_to_classid: dict[str, list[str]] = {}
    for nid, data in g.nodes(data="data"):
        if data and data.node_type == NodeType.CLASS:
            short_to_classid.setdefault(data.name, []).append(nid)
    for nid, data in list(g.nodes(data="data")):
        if not data or data.node_type != NodeType.CLASS:
            continue
        for base in data.extra.get("bases", []):
            short = base.split(".")[-1]
            for parent_id in short_to_classid.get(short, []):
                if parent_id != nid:
                    g.add_edge(
                        nid,
                        parent_id,
                        key=EdgeType.INHERIT.value,
                        edge_type=EdgeType.INHERIT,
                    )

    # INVOKE edges: read source files (only those in skels) and resolve
    if repo_root is not None:
        for sk in skels:
            try:
                file_to_source[sk.file_path] = (repo_root / sk.file_path).read_bytes()
            except OSError:
                continue
        if file_to_source:
            invoke_edges = build_call_edges(file_to_source, file_to_qualnames)
            for caller_id, callee_id, line in invoke_edges:
                if caller_id in g and callee_id in g:
                    g.add_edge(
                        caller_id,
                        callee_id,
                        key=EdgeType.INVOKE.value,
                        edge_type=EdgeType.INVOKE,
                        line=line,
                    )

    return g


def _import_target(import_stmt: str) -> str | None:
    """Best-effort: 'from a.b.c import x' -> 'a/b/c.py'.
    Imperfect — we drop dangling targets after construction.
    """
    s = import_stmt.strip()
    if s.startswith("from "):
        try:
            mod = s.split()[1]
        except IndexError:
            return None
    elif s.startswith("import "):
        mod = s[len("import ") :].split()[0].split(",")[0]
    else:
        return None
    mod = mod.lstrip(".")
    if not mod:
        return None
    return mod.replace(".", "/") + ".py"
