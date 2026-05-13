"""ARISE-style intra-procedural def-use edges. Lazy: only built for drilled functions."""

from __future__ import annotations

from dataclasses import dataclass

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language

from .nodes import EdgeType, NodeData, NodeType, fid_function, fid_statement

_PY_LANGUAGE = get_language("python")


@dataclass
class StmtRecord:
    line: int
    text: str
    defs: set[str]
    uses: set[str]


def _text(n: Node, src: bytes) -> str:
    return src[n.start_byte : n.end_byte].decode("utf-8", errors="replace")


def _collect_idents(n: Node, src: bytes) -> set[str]:
    out: set[str] = set()
    stack = [n]
    while stack:
        x = stack.pop()
        if x.type == "identifier":
            out.add(_text(x, src))
        stack.extend(x.children)
    return out


def _stmt_def_use(n: Node, src: bytes) -> tuple[set[str], set[str]]:
    """Crude SSA-ish: defs = LHS identifiers of assignments, uses = all other idents."""
    defs: set[str] = set()
    if n.type in ("assignment", "augmented_assignment"):
        lhs = n.child_by_field_name("left")
        if lhs is not None:
            defs |= _collect_idents(lhs, src)
        rhs = n.child_by_field_name("right")
        uses = _collect_idents(rhs, src) if rhs else set()
        return defs, uses
    if n.type == "for_statement":
        var = n.child_by_field_name("left")
        if var is not None:
            defs |= _collect_idents(var, src)
        iterable = n.child_by_field_name("right")
        uses = _collect_idents(iterable, src) if iterable else set()
        return defs, uses
    return defs, _collect_idents(n, src)


def materialize_dataflow_for_function(
    g, file_source: bytes, file_path: str, qualname: str, fn_node: Node
) -> int:
    """Adds Statement nodes + DEF_USE edges for one function. Returns #stmts added."""
    body = fn_node.child_by_field_name("body")
    if body is None:
        return 0

    stmts: list[StmtRecord] = []
    for child in body.children:
        if child.type in (
            "function_definition",
            "decorated_definition",
            "class_definition",
        ):
            continue
        defs, uses = _stmt_def_use(child, file_source)
        stmts.append(
            StmtRecord(
                line=child.start_point[0] + 1,
                text=_text(child, file_source).strip().splitlines()[0][:200],
                defs=defs,
                uses=uses,
            )
        )

    fn_id = fid_function(file_path, qualname)
    if fn_id not in g:
        return 0

    # add statement nodes
    stmt_ids: list[str] = []
    for st in stmts:
        sid = fid_statement(file_path, qualname, st.line)
        g.add_node(
            sid,
            data=NodeData(
                node_type=NodeType.STATEMENT,
                name=f"L{st.line}",
                file_path=file_path,
                qualname=qualname,
                start_line=st.line,
                end_line=st.line,
                code=st.text,
                extra={"defs": list(st.defs), "uses": list(st.uses)},
            ),
        )
        g.add_edge(fn_id, sid, key=EdgeType.CONTAIN.value, edge_type=EdgeType.CONTAIN)
        stmt_ids.append(sid)

    # def-use edges: for each stmt's use, link from the most-recent prior stmt that defs it
    last_def: dict[str, str] = {}
    for st, sid in zip(stmts, stmt_ids):
        for u in st.uses:
            if u in last_def and last_def[u] != sid:
                g.add_edge(
                    last_def[u],
                    sid,
                    key=EdgeType.DEF_USE.value,
                    edge_type=EdgeType.DEF_USE,
                    var=u,
                )
        for d in st.defs:
            last_def[d] = sid
    return len(stmts)


def find_function_node(source: bytes, qualname: str) -> Node | None:
    parser = Parser()
    parser.set_language(_PY_LANGUAGE)
    tree = parser.parse(source)
    target_parts = qualname.split(".")

    def walk(node: Node, prefix: list[str]) -> Node | None:
        for child in node.children:
            if child.type == "class_definition":
                name = child.child_by_field_name("name")
                cname = _text(name, source) if name else ""
                body = child.child_by_field_name("body")
                if body:
                    found = walk(body, prefix + [cname])
                    if found is not None:
                        return found
            elif child.type in ("function_definition", "decorated_definition"):
                fn = child
                if fn.type == "decorated_definition":
                    inner = next(
                        (c for c in fn.children if c.type == "function_definition"),
                        None,
                    )
                    if inner is None:
                        continue
                    fn = inner
                name = fn.child_by_field_name("name")
                if name is None:
                    continue
                here = prefix + [_text(name, source)]
                if here == target_parts:
                    return fn
                body = fn.child_by_field_name("body")
                if body:
                    found = walk(body, here)
                    if found is not None:
                        return found
        return None

    return walk(tree.root_node, [])
