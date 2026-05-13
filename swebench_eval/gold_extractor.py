"""Parse SWE-bench gold patches into (file, function, line-range) ground-truth sets.

Critical: cross-validate against LocAgent's `gold_extractor.py` for at least 50
instances before trusting the metrics. The function-level extraction is the
most subtle part — it must locate the enclosing function in the PRE-PATCH
source so it aligns with what we localize on.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language
from unidiff import PatchSet

_PY_LANGUAGE = get_language("python")


@dataclass
class GoldLocation:
    files: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)   # "<file>::<qualname>"
    line_ranges: list[tuple[str, int, int]] = field(default_factory=list)  # (file, start, end)


def _enclosing_qualname(source: bytes, target_line: int) -> str | None:
    """Return the deepest class.method qualname enclosing line `target_line` (1-indexed)."""
    parser = Parser()
    parser.set_language(_PY_LANGUAGE)
    try:
        tree = parser.parse(source)
    except Exception:
        return None

    found: list[str] = []

    def walk(node: Node, prefix: list[str]) -> None:
        for child in node.children:
            sl = child.start_point[0] + 1
            el = child.end_point[0] + 1
            if not (sl <= target_line <= el):
                continue
            if child.type == "class_definition":
                name = child.child_by_field_name("name")
                cname = (
                    source[name.start_byte : name.end_byte].decode("utf-8", errors="replace")
                    if name
                    else ""
                )
                body = child.child_by_field_name("body")
                if body:
                    walk(body, prefix + [cname])
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
                fname = source[name.start_byte : name.end_byte].decode("utf-8", errors="replace")
                here = prefix + [fname]
                found.clear()
                found.append(".".join(here))
                body = fn.child_by_field_name("body")
                if body:
                    walk(body, here)

    walk(tree.root_node, [])
    return found[0] if found else None


def extract_gold_for_instance(
    *,
    patch_text: str,
    repo_path: Path,
    base_commit: str,
) -> GoldLocation:
    """Returns the gold (file, function, line) set for one SWE-bench instance.

    `patch_text` is the unified diff stored in instance["patch"].
    `repo_path` should be checked out at `base_commit` already.
    """
    gold = GoldLocation()
    if not patch_text.strip():
        return gold
    try:
        ps = PatchSet(patch_text)
    except Exception:
        return gold

    for pf in ps:
        # path on the post-patch side ('+++ b/...')
        target = pf.path
        if target.startswith("a/"):
            target = target[2:]
        if target.startswith("b/"):
            target = target[2:]
        if not target.endswith(".py"):
            continue
        gold.files.add(target)

        # Read pre-patch source for enclosing-function lookup
        try:
            pre_source = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_path),
                    "show",
                    f"{base_commit}:{target}",
                ],
                check=False,
                capture_output=True,
            ).stdout
        except Exception:
            pre_source = b""

        for hunk in pf:
            # source-side line range (the lines in the pre-patch file this hunk touches)
            start = hunk.source_start
            length = hunk.source_length or 1
            end = start + length - 1
            gold.line_ranges.append((target, start, end))
            if pre_source:
                # find the enclosing function for the *first* changed line in the hunk
                for line in hunk:
                    if line.is_added or line.is_removed:
                        line_no = line.source_line_no or line.target_line_no
                        if line_no is None:
                            continue
                        qual = _enclosing_qualname(pre_source, line_no)
                        if qual:
                            gold.functions.add(f"{target}::{qual}")
                        break

    return gold
