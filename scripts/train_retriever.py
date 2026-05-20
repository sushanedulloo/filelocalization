"""Fine-tune the bi-encoder on SweLoc.

Usage:
    python scripts/train_retriever.py --sweloc <path-or-hf-id> --output data/embeddings/sweloc_finetuned
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from hybridloc.retrieval.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweloc", default="Salesforce/SweLoc",
                    help="HF dataset id or path to a local arrow/json file")
    ap.add_argument("--output", default="data/embeddings/sweloc_finetuned", type=Path)
    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch-size", default=4, type=int,
                    help="conservative default for 11GB GPUs")
    ap.add_argument("--max-seq-length", default=2048, type=int,
                    help="lower than 8192 to fit on 11GB GPUs")
    args = ap.parse_args()

    print(f"Loading SweLoc from {args.sweloc} ...")
    ds = load_dataset(args.sweloc)
    train_split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    dev_split = ds.get("validation") if hasattr(ds, "get") else (
        ds["validation"] if "validation" in ds else None
    )
    print(f"  {len(train_split)} training examples")

    cfg = TrainConfig(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    out_path = train(train_split, dev_split, cfg)
    print(f"saved fine-tuned bi-encoder to {out_path}")


if __name__ == "__main__":
    main()
