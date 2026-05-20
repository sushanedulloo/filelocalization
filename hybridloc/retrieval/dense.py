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
            # Rank GPUs by free memory, then probe each one in order: actually
            # try to init CUBLAS via a tiny matmul. On shared servers a GPU can
            # report "free" memory but be in a corrupted CUDA-context state
            # (CUBLAS_STATUS_NOT_INITIALIZED). Skip those and try the next.
            candidates: list[tuple[int, int]] = []
            for i in range(torch.cuda.device_count()):
                try:
                    free = torch.cuda.mem_get_info(i)[0]
                except Exception:
                    continue
                if free > 6 * 1024**3:  # need ~6GB headroom for indexing
                    candidates.append((i, free))
            candidates.sort(key=lambda x: -x[1])  # most-free first

            self.device = "cpu"
            for idx, free in candidates:
                try:
                    with torch.cuda.device(idx):
                        a = torch.randn(8, 8, device=f"cuda:{idx}")
                        b = torch.randn(8, 8, device=f"cuda:{idx}")
                        _ = (a @ b).sum().item()  # forces CUBLAS init
                        del a, b
                        torch.cuda.empty_cache()
                    self.device = f"cuda:{idx}"
                    break
                except Exception as e:
                    from ..log import warn
                    warn(f"[dense] cuda:{idx} failed CUBLAS probe ({type(e).__name__}); trying next")
                    continue
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

    def _empty_gpu_cache(self) -> None:
        """Release PyTorch's reserved-but-unallocated GPU memory before indexing."""
        try:
            import torch
            if "cuda" in str(self.device) and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _fallback_to_cpu(self) -> None:
        """Move the model to CPU after a GPU OOM."""
        from ..log import warn
        warn(f"[dense] GPU OOM — falling back to CPU for the rest of indexing")
        self.device = "cpu"
        if self._model is not None:
            try:
                self._model = self._model.to("cpu")  # type: ignore[union-attr]
            except Exception:
                self._model = None  # force reload on CPU

    def _encode_with_oom_retry(self, batch: list[str], batch_size: int) -> np.ndarray:
        """Encode one batch with adaptive shrinking on GPU OOM.

        On OOM: clear cache, halve max_tokens_per_batch by splitting the
        batch in half. If splits still fail at size 1, fall back to CPU.
        """
        try:
            return self.encode(batch, batch_size=batch_size)
        except Exception as e:
            if "out of memory" not in str(e).lower() and "OutOfMemory" not in type(e).__name__:
                raise
            self._empty_gpu_cache()
            if len(batch) == 1:
                # single document still OOMs — switch to CPU and retry
                self._fallback_to_cpu()
                return self.encode(batch, batch_size=1)
            # split the batch in half and recurse
            mid = len(batch) // 2
            left = self._encode_with_oom_retry(batch[:mid], batch_size=max(1, batch_size // 2))
            right = self._encode_with_oom_retry(batch[mid:], batch_size=max(1, batch_size // 2))
            return np.concatenate([left, right], axis=0)

    def index(
        self,
        doc_texts: list[str],
        doc_ids: list[str],
        batch_size: int = 8,
        max_tokens_per_batch: int = 4096,
    ) -> None:
        """Encode documents using token-bounded batching with OOM recovery.

        - Token-bounded batching keeps peak memory bounded regardless of file
          length variance (SweRank §4.3 style).
        - Clears PyTorch's reserved-but-unallocated cache before starting.
        - On OOM mid-batch: halves the batch and retries; falls back to CPU
          if even a single document OOMs.
        """
        if len(doc_texts) != len(doc_ids):
            raise ValueError("doc_texts and doc_ids must align")
        self._doc_ids = list(doc_ids)

        # Free any reserved-but-unallocated GPU memory from earlier stages
        self._empty_gpu_cache()

        chunks: list[np.ndarray] = []
        batch: list[str] = []
        batch_tokens = 0
        for text in doc_texts:
            text_tokens = len(text.split())
            if batch and batch_tokens + text_tokens > max_tokens_per_batch:
                chunks.append(self._encode_with_oom_retry(batch, batch_size=batch_size))
                self._empty_gpu_cache()  # release between batches
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += text_tokens
        if batch:
            chunks.append(self._encode_with_oom_retry(batch, batch_size=batch_size))

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
