"""Gold-extractor sanity test on a fabricated unified diff + pre-patch source."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from swebench_eval.gold_extractor import _enclosing_qualname, extract_gold_for_instance


PRE_SOURCE = b'''\
class Foo:
    def bar(self, x):
        y = x + 1
        return y

    def baz(self):
        return 42

def top():
    return 0
'''


def test_enclosing_qualname_class_method():
    assert _enclosing_qualname(PRE_SOURCE, 3) == "Foo.bar"
    assert _enclosing_qualname(PRE_SOURCE, 7) == "Foo.baz"
    assert _enclosing_qualname(PRE_SOURCE, 10) == "top"


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a tiny git repo with one commit, return (path, sha)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "m.py").write_bytes(PRE_SOURCE)
    subprocess.run(["git", "-C", str(tmp_path), "add", "m.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tmp_path, sha


def test_extract_gold_locates_function_and_lines(fake_repo):
    repo_path, sha = fake_repo
    patch = """\
diff --git a/m.py b/m.py
index 0000000..1111111 100644
--- a/m.py
+++ b/m.py
@@ -2,3 +2,3 @@ class Foo:
     def bar(self, x):
-        y = x + 1
+        y = x + 2
         return y
"""
    gl = extract_gold_for_instance(
        patch_text=patch, repo_path=repo_path, base_commit=sha
    )
    assert "m.py" in gl.files
    assert "m.py::Foo.bar" in gl.functions
    # line range from the unidiff hunk's source side
    assert any(f == "m.py" and s <= 3 <= e for f, s, e in gl.line_ranges)
