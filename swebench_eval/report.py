"""Write the metrics CSV and a markdown comparison table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import Gold, Prediction, aggregate, expected_calibration_error


# Published numbers for the four baselines on SWE-bench Verified (file-level
# unless otherwise noted). Sources: each paper's reported tables. These are
# placeholders we update as we re-verify; the eval script prints
# `(paper-quoted)` next to them so reviewers know they aren't reproduced.
BASELINES = {
    "Agentless": {
        "file_acc@1": None,
        "file_acc@5": 0.768,
        "func_recall@5": None,
        "mrr": None,
        "line_recall@5": None,
    },
    "LocAgent": {
        "file_acc@1": None,
        "file_acc@5": 0.927,    # 92.7% file-level acc with Qwen2.5-Coder-32B
        "func_recall@5": None,
        "mrr": None,
        "line_recall@5": None,
    },
    "SweRank": {
        "file_acc@5": None,
        "func_recall@5": None,
        "mrr": 0.818,           # paper-reported MRR on SWE-bench Verified file-level
    },
    "RepoMem": {
        "file_acc@5": 0.815,    # +4.9 over LocAgent baseline; absolute number from paper Table 2
    },
    "ARISE": {
        "line_recall@5": None,  # paper reports +15 pts Line Recall@1 over SWE-agent
    },
}


def write_report(
    preds: list[Prediction],
    golds: list[Gold],
    *,
    out_csv: Path,
    out_md: Path | None = None,
) -> dict:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = aggregate(preds, golds)
    summary["ece"] = expected_calibration_error(preds, golds)
    summary["n_predictions"] = len(preds)
    summary["n_with_gold_function"] = sum(1 for g in golds if g.functions)

    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    rows: list[dict] = []
    for p in preds:
        rows.append({"instance_id": p.instance_id, "top1_func": p.functions[:1], "n_files": len(p.files)})
    pd.DataFrame(rows).to_csv(out_csv.with_suffix(".per_instance.csv"), index=False)

    if out_md is not None:
        _write_md(summary, out_md)

    return summary


def _write_md(summary: dict, out: Path) -> None:
    lines: list[str] = []
    lines.append("# HybridLoc v2 — SWE-bench Verified results\n")
    lines.append(f"Predictions: {summary.get('n_predictions', 0)}  ")
    lines.append(f"With gold-function set: {summary.get('n_with_gold_function', 0)}\n")

    cols = ["file_acc@1", "file_acc@5", "func_recall@5", "func_acc@5", "mrr", "line_recall@5"]
    headers = ["Method"] + cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for name, vals in BASELINES.items():
        row = [name] + [
            f"{vals[c]:.3f} (paper)" if vals.get(c) is not None else "–"
            for c in cols
        ]
        lines.append("| " + " | ".join(row) + " |")
    ours = ["**HybridLoc v2 (ours)**"] + [
        f"**{summary.get(c, float('nan')):.3f}**" if summary.get(c) == summary.get(c) else "–"
        for c in cols
    ]
    lines.append("| " + " | ".join(ours) + " |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
