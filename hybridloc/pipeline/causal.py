"""GraphLocator-style abductive causal reasoning. Wraps the Think-High prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.nim_client import NIMClient, ReasoningMode

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


@dataclass
class CausalChain:
    function_key: str    # "<file_path>::<qualname>"
    chain: list[str] = field(default_factory=list)
    score: float = 0.0   # 0-10 from the LLM


@dataclass
class CausalUpdate:
    chains: list[CausalChain] = field(default_factory=list)
    next_to_explore: list[str] = field(default_factory=list)


class AbductiveReasoner:
    def __init__(self, nim: NIMClient):
        self.nim = nim
        self._tpl = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "prompts"
            / "causal_chain.txt"
        ).read_text()

    def update_chains(
        self,
        *,
        issue: str,
        symptoms_text: str,
        cig_mermaid: str,
        candidates: list[tuple[str, str]],   # (function_key, code)
    ) -> CausalUpdate:
        if not candidates:
            return CausalUpdate()
        candidate_block = "\n\n".join(
            f"### {key}\n```python\n{code[:2000]}\n```" for key, code in candidates
        )
        prompt = self._tpl.format(
            issue=issue.strip()[:6000],
            symptoms=symptoms_text[:2000],
            cig_mermaid=cig_mermaid[:4000],
            candidates=candidate_block[:60_000],
        )
        resp = self.nim.complete(
            prompt,
            mode=ReasoningMode.THINK_HIGH,
            json_schema={"type": "object"},
            temperature=0.0,
        )
        from ..log import info
        update = _parse(resp.text)
        for ch in update.chains:
            info(f"[Causal] {ch.function_key} score={ch.score:.1f} chain={' → '.join(ch.chain[:4])}")
        info(f"[Causal] next_to_explore: {update.next_to_explore[:5]}")
        return update


def _parse(raw: str) -> CausalUpdate:
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return CausalUpdate()
    if not isinstance(obj, dict):
        return CausalUpdate()
    chains_raw = obj.get("chains") or []
    chains: list[CausalChain] = []
    for c in chains_raw:
        if not isinstance(c, dict):
            continue
        try:
            chains.append(
                CausalChain(
                    function_key=str(c.get("function", "")),
                    chain=[str(x) for x in (c.get("chain") or [])],
                    score=float(c.get("score", 0.0)),
                )
            )
        except (ValueError, TypeError):
            continue
    nxt = [str(x) for x in (obj.get("next_to_explore") or [])]
    return CausalUpdate(chains=chains, next_to_explore=nxt)
