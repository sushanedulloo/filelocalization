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
        import os
        self.model_name = model_name
        self.finetuned_path = Path(finetuned_path) if finetuned_path else None
        # respect HYBRIDLOC_EMBED_DEVICE env var; auto-pick GPU with most free memory, else CPU
        import torch
        if device:
            self.device = device
        elif os.environ.get("HYBRIDLOC_EMBED_DEVICE"):
            self.device = os.environ["HYBRIDLOC_EMBED_DEVICE"]
        elif torch.cuda.is_available():
            best_gpu, best_free = 0, 0
            for i in range(torch.cuda.device_count()):
                free = torch.cuda.mem_get_info(i)[0]
                if free > best_free:
                    best_free, best_gpu = free, i
            # only use GPU if at least 6GB free (model needs ~550MB but indexing needs more)
            self.device = f"cuda:{best_gpu}" if best_free > 6 * 1024**3 else "cpu"
        else:
            self.device = "cpu"
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

    def index(
        self,
        doc_texts: list[str],
        doc_ids: list[str],
        batch_size: int = 8,
        max_tokens_per_batch: int = 8192,
    ) -> None:
        """Encode documents using token-bounded batching (SweRank §4.3 style).

        File text length varies a lot — batching by file count OOMs on long files.
        We group documents until the cumulative token estimate hits the budget,
        then flush the batch. This keeps peak GPU memory bounded regardless of
        how many short or long files the repo has.
        """
        if len(doc_texts) != len(doc_ids):
            raise ValueError("doc_texts and doc_ids must align")
        self._doc_ids = list(doc_ids)

        chunks: list[np.ndarray] = []
        batch: list[str] = []
        batch_tokens = 0
        for text in doc_texts:
            # cheap whitespace-token estimate; bi-encoder tokenizer typically
            # produces 1.3-1.5x this count, so 8192 word-tokens ≈ 12k model tokens
            text_tokens = len(text.split())
            if batch and batch_tokens + text_tokens > max_tokens_per_batch:
                chunks.append(self.encode(batch, batch_size=batch_size))
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += text_tokens
        if batch:
            chunks.append(self.encode(batch, batch_size=batch_size))

        self._doc_emb = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0,), dtype=np.float32)
        )

    def query(self, q: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._doc_emb is None:
            raise RuntimeError("call .index(...) before .query(...)")
        q_emb = self.encode([q])[0]
        scores = self._doc_emb @ q_emb  # both unit-normed -> cosine
        order = np.argsort(-scores)[:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in order]
