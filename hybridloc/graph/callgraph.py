"""Static call graph extraction.

Strategy: Tree-sitter walks call sites; Jedi (when available) resolves them
across files. Falls back to identifier-match within imports of the same file.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tree_sitter import Node
from tree_sitter_languages import get_language
from tree_sitter import Parser

from .nodes import fid_function

_PY_LANGUAGE = get_language("python")


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def extract_calls_in_function(
    file_source: bytes, fn_node: Node
) -> list[tuple[str, int]]:
    """Return [(callee_text, line), ...] for call sites inside `fn_node`."""
    out: list[tuple[str, int]] = []
    stack = [fn_node]
    while stack:
        n = stack.pop()
        if n.type == "call":
            callee = n.child_by_field_name("function")
            if callee is not None:
                out.append((_text(callee, file_source), n.start_point[0] + 1))
        # don't descend into nested function defs to avoid attributing inner
        # calls to outer functions; that's a separate function node
        if n.type in ("function_definition", "decorated_definition") and n is not fn_node:
            continue
        stack.extend(n.children)
    return out


def parse_module_calls(source: bytes) -> dict[str, list[tuple[str, int]]]:
    """{fn_qualname: [(callee_text, line), ...]} for all functions in the module."""
    parser = Parser()
    parser.set_language(_PY_LANGUAGE)
    tree = parser.parse(source)
    out: dict[str, list[tuple[str, int]]] = {}

    def walk(node: Node, qualprefix: str) -> None:
        for child in node.children:
            if child.type == "class_definition":
                name = child.child_by_field_name("name")
                cname = _text(name, source) if name else "<anon>"
                body = child.child_by_field_name("body")
                if body:
                    walk(body, qualprefix + cname + ".")
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
                fn_name = _text(name, source)
                qualname = qualprefix + fn_name
                out[qualname] = extract_calls_in_function(source, fn)
                # recurse to find nested defs (rare)
                body = fn.child_by_field_name("body")
                if body:
                    walk(body, qualname + ".")

    walk(tree.root_node, "")
    return out


def resolve_callee_text(
    callee_text: str,
    *,
    same_file_qualnames: set[str],
    name_to_files: dict[str, list[str]],
    current_file: str,
) -> str | None:
    """Heuristic resolution: returns a node-id for the callee if we can name it."""
    name = callee_text.split(".")[-1].strip()
    if not name:
        return None
    # 1. Same-file match (function or method)
    for q in same_file_qualnames:
        if q == name or q.endswith("." + name):
            return fid_function(current_file, q)
    # 2. Cross-file by exact short name (best effort)
    paths = name_to_files.get(name, [])
    if len(paths) == 1:
        # only resolve when unambiguous to keep precision
        return fid_function(paths[0], name)
    return None


def build_call_edges(
    file_to_source: dict[str, bytes],
    file_to_qualnames: dict[str, set[str]],
) -> list[tuple[str, str, int]]:
    """Returns list of (caller_node_id, callee_node_id, line)."""
    # build index: short-name -> [files]
    name_to_files: dict[str, list[str]] = defaultdict(list)
    for path, qualnames in file_to_qualnames.items():
        for q in qualnames:
            short = q.split(".")[-1]
            name_to_files[short].append(path)

    edges: list[tuple[str, str, int]] = []
    for path, src in file_to_source.items():
        try:
            calls = parse_module_calls(src)
        except Exception:
            continue
        same = file_to_qualnames.get(path, set())
        for caller_qn, sites in calls.items():
            caller_id = fid_function(path, caller_qn)
            for callee_text, line in sites:
                callee_id = resolve_callee_text(
                    callee_text,
                    same_file_qualnames=same,
                    name_to_files=name_to_files,
                    current_file=path,
                )
                if callee_id and callee_id != caller_id:
                    edges.append((caller_id, callee_id, line))
    return edges
