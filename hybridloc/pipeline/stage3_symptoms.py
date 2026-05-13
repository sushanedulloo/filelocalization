"""Stage 3: symptom extraction from the issue text -> Symptom nodes + seed scores."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm.nim_client import NIMClient, ReasoningMode


@dataclass
class StackFrame:
    file: str
    func: str
    line: int


@dataclass
class Symptoms:
    exception_types: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)
    stack_frames: list[StackFrame] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    api_calls_named: list[str] = field(default_factory=list)

    def keywords(self) -> list[str]:
        out = list(self.exception_types) + list(self.behaviors) + list(self.api_calls_named)
        return [k for k in out if k]


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


class SymptomExtractor:
    def __init__(self, nim: NIMClient):
        self.nim = nim
        self._tpl = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "prompts"
            / "symptom_extract.txt"
        ).read_text()

    def extract(self, issue: str) -> Symptoms:
        from ..log import info
        prompt = self._tpl.format(issue=issue.strip()[:8000])
        resp = self.nim.complete(
            prompt,
            mode=ReasoningMode.NON_THINK,
            json_schema={"type": "object"},
            temperature=0.0,
        )
        symptoms = _parse(resp.text)
        info(f"[Stage 3] exception_types: {symptoms.exception_types}")
        info(f"[Stage 3] error_messages:  {symptoms.error_messages[:3]}")
        info(f"[Stage 3] stack_frames:    {[(sf.file, sf.func, sf.line) for sf in symptoms.stack_frames]}")
        info(f"[Stage 3] behaviors:       {symptoms.behaviors}")
        info(f"[Stage 3] api_calls_named: {symptoms.api_calls_named}")
        return symptoms


def _parse(raw: str) -> Symptoms:
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return Symptoms()
    if not isinstance(obj, dict):
        return Symptoms()
    out = Symptoms(
        exception_types=[str(x) for x in obj.get("exception_types") or []],
        error_messages=[str(x) for x in obj.get("error_messages") or []],
        behaviors=[str(x) for x in obj.get("behaviors") or []],
        api_calls_named=[str(x) for x in obj.get("api_calls_named") or []],
    )
    for sf in obj.get("stack_frames") or []:
        if isinstance(sf, dict) and "file" in sf:
            try:
                out.stack_frames.append(
                    StackFrame(
                        file=str(sf["file"]),
                        func=str(sf.get("func", "")),
                        line=int(sf.get("line", 0)),
                    )
                )
            except (ValueError, TypeError):
                continue
    return out
