"""Fine-tune CodeRankEmbed on SweLoc + repo-mined pairs.

3x RTX 2080 Ti, fp16 (Turing has no bf16), MultipleNegativesRankingLoss + hard negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    model_name: str = "nomic-ai/CodeRankEmbed"
    output_dir: Path = Path("data/embeddings/sweloc_finetuned")
    batch_size: int = 16
    grad_accum: int = 4
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 8192
    seed: int = 42
    fp16: bool = True
    eval_every: int = 1000


def to_examples(rows) -> list[InputExample]:
    """Each row: {issue: str, positive: str, negatives: list[str]}.
    Returns InputExamples for MultipleNegativesRankingLoss with optional hard negs.
    """
    out: list[InputExample] = []
    for r in rows:
        anchor = r["issue"]
        pos = r["positive"]
        # MNR loss treats each example as (anchor, positive); other examples in
        # the batch act as in-batch negatives. Hard negs we add as separate
        # texts so they appear as in-batch negatives too.
        out.append(InputExample(texts=[anchor, pos]))
        for neg in (r.get("negatives") or [])[:1]:
            out.append(InputExample(texts=[anchor, neg]))
    return out


def train(
    train_dataset: Dataset,
    dev_dataset: Dataset | None = None,
    config: TrainConfig | None = None,
) -> Path:
    cfg = config or TrainConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    model = SentenceTransformer(cfg.model_name, trust_remote_code=True)
    model.max_seq_length = cfg.max_seq_length

    train_examples = to_examples(train_dataset)
    loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=cfg.batch_size,
        collate_fn=model.smart_batching_collate,
    )
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(loader) * cfg.epochs * cfg.warmup_ratio)
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=cfg.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": cfg.learning_rate},
        use_amp=cfg.fp16,
        output_path=str(cfg.output_dir),
        save_best_model=True,
        checkpoint_save_steps=cfg.eval_every,
        checkpoint_save_total_limit=2,
    )
    return cfg.output_dir
