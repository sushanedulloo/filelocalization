"""Tree-sitter skeleton extraction smoke test on a tiny fixture."""

from __future__ import annotations

from pathlib import Path

from hybridloc.parsing.skeleton import PythonSkeletonExtractor


FIXTURE = '''\
"""mod docstring"""
import os
from foo.bar import baz

class Animal(Base):
    """An animal."""
    def speak(self, x):
        return x

def free_function(a, b=1):
    """does a thing"""
    return a + b
'''


def test_extracts_classes_methods_and_functions(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text(FIXTURE)
    sk = PythonSkeletonExtractor().extract(f, tmp_path)
    assert sk is not None
    assert sk.file_path == "mod.py"
    assert any("from foo.bar" in i for i in sk.imports)
    assert len(sk.classes) == 1
    cls = sk.classes[0]
    assert cls.name == "Animal"
    assert "Base" in cls.bases
    assert any(m.name == "speak" for m in cls.methods)
    assert any(f.name == "free_function" for f in sk.functions)


def test_compact_text_is_short(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text(FIXTURE)
    sk = PythonSkeletonExtractor().extract(f, tmp_path)
    assert sk is not None
    text = sk.as_compact_text()
    assert "FILE: mod.py" in text
    assert "class Animal" in text
    assert "def speak" in text
