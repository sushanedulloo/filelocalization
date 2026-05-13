"""Dense retriever wrapping CodeRankEmbed (or our SweLoc fine-tune)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DenseRetriever:
    """Lazy-loaded wrapper around sentence-transformers.

    Pretrained backbone: nomic-ai/CodeRankEmbed (137M, 8192 ctx).
    After Stage 5 fine-tune, swap in `data/embeddings/sweloc_finetuned/`.
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/CodeRankEmbed",
        finetuned_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.finetuned_path = Path(finetuned_path) if finetuned_path else None
        self.device = device
        self._model = None  # type: ignore[assignment]
        self._doc_ids: list[str] = []
        self._doc_emb: np.ndarray | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            target = (
                str(self.finetuned_path)
                if self.finetuned_path and self.finetuned_path.exists()
                else self.model_name
            )
            self._model = SentenceTransformer(target, device=self.device, trust_remote_code=True)
        return self._model

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        m = self._load()
        embs = m.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(embs, dtype=np.float32)

    def index(self, doc_texts: list[str], doc_ids: list[str], batch_size: int = 32) -> None:
        if len(doc_texts) != len(doc_ids):
            raise ValueError("doc_texts and doc_ids must align")
        self._doc_ids = list(doc_ids)
        self._doc_emb = self.encode(doc_texts, batch_size=batch_size)

    def query(self, q: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._doc_emb is None:
            raise RuntimeError("call .index(...) before .query(...)")
        q_emb = self.encode([q])[0]
        scores = self._doc_emb @ q_emb  # both unit-normed -> cosine
        order = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in order]
