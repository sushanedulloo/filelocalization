"""Tree-sitter skeleton extraction. Python only in v1 (per scope §13)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language


@dataclass
class FunctionInfo:
    name: str
    qualname: str           # ClassName.method or function
    start_line: int         # 1-indexed inclusive
    end_line: int
    docstring: str = ""
    signature: str = ""


@dataclass
class ClassInfo:
    name: str
    start_line: int
    end_line: int
    bases: list[str] = field(default_factory=list)
    docstring: str = ""
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class Skeleton:
    file_path: str          # repo-relative, posix
    language: str
    imports: list[str] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)  # module-level only
    summary: str = ""       # 1-line NL summary, filled later by NIM
    n_lines: int = 0

    def as_compact_text(self) -> str:
        """Compact textual form fed to the BM25 retriever and the LLM ranker."""
        lines = [f"FILE: {self.file_path}"]
        if self.summary:
            lines.append(f"SUMMARY: {self.summary}")
        if self.imports:
            lines.append("IMPORTS: " + ", ".join(self.imports[:50]))
        for c in self.classes:
            base = f"({', '.join(c.bases)})" if c.bases else ""
            lines.append(f"class {c.name}{base}")
            if c.docstring:
                lines.append(f"  \"\"\"{c.docstring[:120]}\"\"\"")
            for m in c.methods:
                lines.append(f"  def {m.signature or m.name}")
                if m.docstring:
                    lines.append(f"    \"\"\"{m.docstring[:120]}\"\"\"")
        for f in self.functions:
            lines.append(f"def {f.signature or f.name}")
            if f.docstring:
                lines.append(f"  \"\"\"{f.docstring[:120]}\"\"\"")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Skeleton":
        d = json.loads(s)
        d["classes"] = [
            ClassInfo(**{**c, "methods": [FunctionInfo(**m) for m in c["methods"]]})
            for c in d["classes"]
        ]
        d["functions"] = [FunctionInfo(**f) for f in d["functions"]]
        return cls(**d)


# ---------------- parser ----------------

_PY_LANGUAGE = get_language("python")


class PythonSkeletonExtractor:
    def __init__(self) -> None:
        self.parser = Parser()
        self.parser.set_language(_PY_LANGUAGE)

    def extract(self, file_path: Path, repo_root: Path) -> Skeleton | None:
        try:
            source = file_path.read_bytes()
        except (OSError, UnicodeDecodeError):
            return None
        if not source:
            return None
        try:
            tree = self.parser.parse(source)
        except Exception:
            return None
        rel = file_path.relative_to(repo_root).as_posix()
        sk = Skeleton(
            file_path=rel,
            language="python",
            n_lines=source.count(b"\n") + 1,
        )
        self._walk(tree.root_node, source, sk)
        return sk

    def _walk(self, root: Node, source: bytes, sk: Skeleton) -> None:
        for node in root.children:
            if node.type in ("import_statement", "import_from_statement"):
                sk.imports.append(_text(node, source).strip())
            elif node.type == "class_definition":
                sk.classes.append(self._class(node, source))
            elif node.type in ("function_definition", "decorated_definition"):
                fn = self._function(node, source, qualprefix="")
                if fn:
                    sk.functions.append(fn)

    def _class(self, node: Node, source: bytes) -> ClassInfo:
        name_node = node.child_by_field_name("name")
        name = _text(name_node, source) if name_node else "<anon>"
        bases: list[str] = []
        sup = node.child_by_field_name("superclasses")
        if sup is not None:
            for c in sup.children:
                if c.type in ("identifier", "attribute"):
                    bases.append(_text(c, source))
        body = node.child_by_field_name("body")
        ci = ClassInfo(
            name=name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            bases=bases,
            docstring=_extract_docstring(body, source) if body else "",
        )
        if body is not None:
            for child in body.children:
                if child.type in ("function_definition", "decorated_definition"):
                    fn = self._function(child, source, qualprefix=name + ".")
                    if fn:
                        ci.methods.append(fn)
        return ci

    def _function(
        self, node: Node, source: bytes, qualprefix: str
    ) -> FunctionInfo | None:
        if node.type == "decorated_definition":
            inner = next(
                (c for c in node.children if c.type == "function_definition"),
                None,
            )
            if inner is None:
                return None
            node = inner
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = _text(name_node, source)
        params_node = node.child_by_field_name("parameters")
        params = _text(params_node, source) if params_node else "()"
        signature = f"{name}{params}"
        body = node.child_by_field_name("body")
        return FunctionInfo(
            name=name,
            qualname=qualprefix + name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=_extract_docstring(body, source) if body else "",
            signature=signature,
        )


def _text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_docstring(body: Node, source: bytes) -> str:
    if body is None:
        return ""
    for child in body.children:
        if child.type == "expression_statement":
            inner = child.children[0] if child.children else None
            if inner and inner.type == "string":
                raw = _text(inner, source).strip()
                # strip leading/trailing quotes (handles """ ''' " ' )
                for q in ('"""', "'''", '"', "'"):
                    if raw.startswith(q) and raw.endswith(q):
                        lines = raw[len(q) : -len(q)].strip().splitlines()
                        return lines[0][:200] if lines else ""
                return raw[:200]
        # only inspect the first statement
        return ""
    return ""


# ---------------- repo-level ----------------

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "build", "dist", ".tox"}


def iter_python_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in repo_root.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def build_repo_skeleton(
    repo_root: Path,
    *,
    max_workers: int = 8,
) -> list[Skeleton]:
    repo_root = repo_root.resolve()
    files = iter_python_files(repo_root)
    extractor = PythonSkeletonExtractor()
    skels: list[Skeleton] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(extractor.extract, f, repo_root): f for f in files}
        for fut in as_completed(futures):
            sk = fut.result()
            if sk is not None:
                skels.append(sk)
    skels.sort(key=lambda s: s.file_path)
    return skels


def save_skeletons(skels: list[Skeleton], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in skels:
            f.write(s.to_json() + "\n")


def load_skeletons(path: Path) -> list[Skeleton]:
    with path.open() as f:
        return [Skeleton.from_json(line) for line in f if line.strip()]
