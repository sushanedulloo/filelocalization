"""Acc@k / Recall@k / MRR / Line Recall@k correctness on hand-built fixtures."""

from __future__ import annotations

from swebench_eval.metrics import (
    Gold,
    Prediction,
    file_acc_at_k,
    function_acc_at_k,
    function_recall_at_k,
    line_recall_at_k,
    mrr,
)


def _gold():
    return Gold(
        instance_id="x",
        files={"a.py", "b.py"},
        functions={"a.py::F", "b.py::G"},
        line_ranges=[("a.py", 10, 20), ("b.py", 50, 60)],
    )


def test_file_acc_at_k_hit_first():
    p = Prediction("x", files=["a.py", "z.py"], functions=[], lines=[], confidences=[])
    assert file_acc_at_k(p, _gold(), 1) == 1.0


def test_file_acc_at_k_miss_at_1_hit_at_3():
    p = Prediction("x", files=["q.py", "r.py", "a.py"], functions=[], lines=[], confidences=[])
    assert file_acc_at_k(p, _gold(), 1) == 0.0
    assert file_acc_at_k(p, _gold(), 3) == 1.0


def test_function_recall_partial():
    p = Prediction("x", files=[], functions=["a.py::F", "z.py::Q"], lines=[], confidences=[])
    assert function_recall_at_k(p, _gold(), 5) == 0.5  # 1 of 2 gold


def test_function_acc_at_k():
    p = Prediction("x", files=[], functions=["z.py::Q", "a.py::F"], lines=[], confidences=[])
    assert function_acc_at_k(p, _gold(), 1) == 0.0
    assert function_acc_at_k(p, _gold(), 2) == 1.0


def test_mrr():
    p = Prediction("x", files=[], functions=["z.py::Q", "a.py::F"], lines=[], confidences=[])
    assert mrr(p, _gold()) == 0.5  # rank 2


def test_line_recall_within_tolerance():
    p = Prediction(
        "x",
        files=[],
        functions=[],
        lines=[("a.py", 25, 30)],   # gold is 10-20, off by 5 -> within tolerance=10
        confidences=[],
    )
    assert line_recall_at_k(p, _gold(), 5, tolerance=10) == 0.5  # 1 of 2 gold


def test_line_recall_outside_tolerance():
    p = Prediction(
        "x",
        files=[],
        functions=[],
        lines=[("a.py", 100, 110)],   # too far
        confidences=[],
    )
    assert line_recall_at_k(p, _gold(), 5, tolerance=10) == 0.0
