"""Node + edge type definitions for the heterogeneous code graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    STATEMENT = "statement"
    SYMPTOM = "symptom"
    CONCEPT = "concept"
    COMMIT = "commit"


class EdgeType(str, Enum):
    CONTAIN = "contain"
    IMPORT = "import"
    INVOKE = "invoke"
    INHERIT = "inherit"
    DEF_USE = "def_use"
    CONCEPT_OF = "concept_of"
    EVOLVED_BY = "evolved_by"
    CO_EVOLVED = "co_evolved"
    SYMPTOM_OF = "symptom_of"


def fid_dir(path: str) -> str:
    return f"dir::{path}"


def fid_file(path: str) -> str:
    return f"file::{path}"


def fid_class(path: str, qualname: str) -> str:
    return f"class::{path}::{qualname}"


def fid_function(path: str, qualname: str) -> str:
    return f"func::{path}::{qualname}"


def fid_statement(path: str, qualname: str, line: int) -> str:
    return f"stmt::{path}::{qualname}::{line}"


def fid_concept(label: str) -> str:
    return f"concept::{label}"


def fid_commit(sha: str) -> str:
    return f"commit::{sha}"


def fid_symptom(text: str) -> str:
    return f"symptom::{text}"


@dataclass
class NodeData:
    """The 'data' dict attached to each NetworkX node."""

    node_type: NodeType
    name: str = ""
    file_path: str = ""
    qualname: str = ""
    start_line: int = 0
    end_line: int = 0
    code: str = ""
    docstring: str = ""
    extra: dict = field(default_factory=dict)
