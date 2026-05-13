"""BM25 over per-file skeleton text."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase identifier tokens, split CamelCase + snake_case to enrich the vocab."""
    raw = _TOKEN_RE.findall(text)
    out: list[str] = []
    for tok in raw:
        out.append(tok.lower())
        # snake_case -> parts
        if "_" in tok:
            out.extend(p.lower() for p in tok.split("_") if p)
        # CamelCase -> parts
        camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", tok)
        if len(camel) > 1:
            out.extend(c.lower() for c in camel)
    return out


class BM25Retriever:
    def __init__(self, doc_texts: list[str], doc_ids: list[str]):
        if len(doc_texts) != len(doc_ids):
            raise ValueError("doc_texts and doc_ids must align")
        self.doc_ids = doc_ids
        self._tokens = [tokenize(t) for t in doc_texts]
        self._bm25 = BM25Okapi(self._tokens)

    def query(self, q: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_toks = tokenize(q)
        if not q_toks:
            return []
        scores = self._bm25.get_scores(q_toks)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in order if scores[i] > 0]
