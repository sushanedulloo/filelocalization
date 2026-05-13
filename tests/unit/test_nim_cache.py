"""Unit tests for the NIM client cache layer.

These do NOT hit the network. They construct an NIMClient with a fake config,
short-circuit the network call, and verify cache semantics + key stability.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hybridloc.llm.nim_client import (
    NIMClient,
    NIMConfig,
    NIMResponse,
    ReasoningMode,
    _cache_key,
)


@pytest.fixture
def client(tmp_path: Path) -> NIMClient:
    cfg = NIMConfig(
        api_key="test-key",
        cache_dir=tmp_path / "nim_cache",
        max_concurrency=4,
    )
    return NIMClient(cfg)


def test_cache_key_is_stable():
    k1 = _cache_key(
        model="m",
        mode=ReasoningMode.NON_THINK,
        messages=[{"role": "user", "content": "hi"}],
        response_format=None,
        temperature=0.0,
        seed=0,
    )
    k2 = _cache_key(
        model="m",
        mode=ReasoningMode.NON_THINK,
        messages=[{"role": "user", "content": "hi"}],
        response_format=None,
        temperature=0.0,
        seed=0,
    )
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex


def test_cache_key_changes_with_mode():
    base = dict(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        response_format=None,
        temperature=0.0,
        seed=0,
    )
    a = _cache_key(mode=ReasoningMode.NON_THINK, **base)
    b = _cache_key(mode=ReasoningMode.THINK_HIGH, **base)
    assert a != b


def test_second_call_hits_cache(client: NIMClient):
    fake_call = lambda **_: ("hello", None)  # noqa: E731

    with patch.object(client, "_call_sync", side_effect=fake_call) as m:
        r1 = client.complete("say hi")
        r2 = client.complete("say hi")

    assert r1.cached is False
    assert r2.cached is True
    assert r2.text == "hello"
    assert m.call_count == 1, "second call must not hit the network"


def test_cache_miss_for_different_prompts(client: NIMClient):
    fake_call = lambda **_: ("ok", None)  # noqa: E731
    with patch.object(client, "_call_sync", side_effect=fake_call) as m:
        client.complete("prompt A")
        client.complete("prompt B")
    assert m.call_count == 2


def test_cache_miss_for_different_modes(client: NIMClient):
    fake_call = lambda **_: ("ok", None)  # noqa: E731
    with patch.object(client, "_call_sync", side_effect=fake_call) as m:
        client.complete("same prompt", mode=ReasoningMode.NON_THINK)
        client.complete("same prompt", mode=ReasoningMode.THINK_HIGH)
    assert m.call_count == 2


def test_extra_body_for_think_modes(client: NIMClient):
    # non-think always empty
    assert client._extra_body(ReasoningMode.NON_THINK) == {}

def test_extra_body_think_high_native_reasoning(tmp_path):
    # R1 reasons natively — no chat_template_kwargs needed
    cfg = NIMConfig(api_key="x", model="deepseek-ai/deepseek-r1", cache_dir=tmp_path)
    c = NIMClient(cfg)
    assert c._extra_body(ReasoningMode.THINK_HIGH) == {}

def test_extra_body_think_high_v4_flash(tmp_path):
    # V4 Flash needs the chat_template_kwargs flag
    cfg = NIMConfig(api_key="x", model="deepseek-ai/deepseek-v4-flash", cache_dir=tmp_path)
    c = NIMClient(cfg)
    assert c._extra_body(ReasoningMode.THINK_HIGH) == {
        "chat_template_kwargs": {"enable_thinking": True, "thinking": True}
    }


def test_response_dataclass_roundtrip():
    r = NIMResponse(text="x", reasoning=None, cached=False, latency_s=0.1, cache_key="k")
    d = r.as_dict()
    assert d["text"] == "x"
    assert d["cache_key"] == "k"
