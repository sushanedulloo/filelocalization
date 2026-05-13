"""Localization metrics: file Acc@k, function Recall@k, line Recall@k (±tol), MRR, ECE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    instance_id: str
    files: list[str]                    # ranked, deduped
    functions: list[str]                # ranked: "<file>::<qualname>"
    lines: list[tuple[str, int, int]]   # ranked: (file, start, end)
    confidences: list[float]            # parallel with `functions`


@dataclass
class Gold:
    instance_id: str
    files: set[str]
    functions: set[str]
    line_ranges: list[tuple[str, int, int]]   # may be many


def file_acc_at_k(p: Prediction, g: Gold, k: int) -> float:
    if not g.files:
        return float("nan")
    hit = any(f in g.files for f in p.files[:k])
    return 1.0 if hit else 0.0


def function_recall_at_k(p: Prediction, g: Gold, k: int) -> float:
    if not g.functions:
        return float("nan")
    pred_set = set(p.functions[:k])
    return len(pred_set & g.functions) / len(g.functions)


def function_acc_at_k(p: Prediction, g: Gold, k: int) -> float:
    if not g.functions:
        return float("nan")
    return 1.0 if any(f in g.functions for f in p.functions[:k]) else 0.0


def mrr(p: Prediction, g: Gold) -> float:
    if not g.functions:
        return float("nan")
    for i, f in enumerate(p.functions, start=1):
        if f in g.functions:
            return 1.0 / i
    return 0.0


def line_recall_at_k(
    p: Prediction, g: Gold, k: int, *, tolerance: int = 10
) -> float:
    if not g.line_ranges:
        return float("nan")
    matched = 0
    for gf, gs, ge in g.line_ranges:
        for pf, ps, pe in p.lines[:k]:
            if pf != gf:
                continue
            if pe + tolerance < gs or ps - tolerance > ge:
                continue
            matched += 1
            break
    return matched / len(g.line_ranges)


def aggregate(
    preds: list[Prediction], golds: list[Gold]
) -> dict[str, float]:
    """Return mean of per-instance metrics, ignoring NaN (no-gold) instances."""
    by_id = {g.instance_id: g for g in golds}
    rows: dict[str, list[float]] = {
        "file_acc@1": [],
        "file_acc@3": [],
        "file_acc@5": [],
        "func_recall@1": [],
        "func_recall@3": [],
        "func_recall@5": [],
        "func_acc@1": [],
        "func_acc@5": [],
        "mrr": [],
        "line_recall@5": [],
    }
    for p in preds:
        g = by_id.get(p.instance_id)
        if g is None:
            continue
        rows["file_acc@1"].append(file_acc_at_k(p, g, 1))
        rows["file_acc@3"].append(file_acc_at_k(p, g, 3))
        rows["file_acc@5"].append(file_acc_at_k(p, g, 5))
        rows["func_recall@1"].append(function_recall_at_k(p, g, 1))
        rows["func_recall@3"].append(function_recall_at_k(p, g, 3))
        rows["func_recall@5"].append(function_recall_at_k(p, g, 5))
        rows["func_acc@1"].append(function_acc_at_k(p, g, 1))
        rows["func_acc@5"].append(function_acc_at_k(p, g, 5))
        rows["mrr"].append(mrr(p, g))
        rows["line_recall@5"].append(line_recall_at_k(p, g, 5))

    out: dict[str, float] = {}
    for k, vs in rows.items():
        arr = np.asarray(vs, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        out[k] = float(arr.mean()) if arr.size else float("nan")
    return out


def expected_calibration_error(
    preds: list[Prediction], golds: list[Gold], n_bins: int = 10
) -> float:
    """ECE for confidence vs whether top-1 hits the gold function set."""
    by_id = {g.instance_id: g for g in golds}
    confidences: list[float] = []
    correct: list[int] = []
    for p in preds:
        g = by_id.get(p.instance_id)
        if g is None or not g.functions or not p.functions:
            continue
        confidences.append(p.confidences[0] if p.confidences else 0.0)
        correct.append(1 if p.functions[0] in g.functions else 0)
    if not confidences:
        return float("nan")
    confs = np.asarray(confidences)
    corr = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = corr[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)
