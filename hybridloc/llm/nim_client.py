"""
NVIDIA NIM client for DeepSeek-V4-Flash.

Three reasoning modes (Non-think / Think-High / Think-Max), an OpenAI-compatible
endpoint, and a content-addressed on-disk cache so repeated runs are free.

The `chat_template_kwargs={"enable_thinking": True, "thinking": True}` payload
is required by NIM to actually return reasoning tokens in the Think modes —
this is the gotcha logged in HybridLoc_v2_plan.md §9.

Synchronous .complete() and async .acomplete() / .acomplete_many() are both
exposed; offline batched passes (e.g. the 50k per-function summary pass for
RepoLens) should use acomplete_many with a Semaphore to stay polite.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()


class ReasoningMode(str, Enum):
    NON_THINK = "non-think"
    THINK_HIGH = "think-high"
    THINK_MAX = "think-max"


@dataclass(frozen=True)
class NIMConfig:
    api_key: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "deepseek-ai/deepseek-v4-flash"
    cache_dir: Path = field(default_factory=lambda: Path("data/nim_cache"))
    max_concurrency: int = 3
    request_timeout: float = 300.0

    @classmethod
    def from_env(cls) -> "NIMConfig":
        api_key = os.environ.get("NIM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "NIM_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        cache_dir = Path(
            os.environ.get("HYBRIDLOC_CACHE_DIR") or "data/nim_cache"
        )
        return cls(
            api_key=api_key,
            base_url=os.environ.get("NIM_BASE_URL", cls.base_url),
            model=os.environ.get("NIM_MODEL", cls.model),
            cache_dir=cache_dir,
        )


_THINK_KWARGS = {"enable_thinking": True, "thinking": True}

# Models that reason natively (no special kwargs needed; thinking happens automatically)
_NATIVE_REASONING_MODELS = {"deepseek-r1", "deepseek-ai/deepseek-r1"}

# NIM free tier: 40 RPM → enforce 38 RPM to stay safely under the limit.
# MIN_INTERVAL = 60 / 38 ≈ 1.58 s between non-cached calls.
_RPM_LIMIT = 38
_MIN_INTERVAL = 60.0 / _RPM_LIMIT   # ~1.58 s


def _cache_key(
    *,
    model: str,
    mode: ReasoningMode,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
    temperature: float,
    seed: int | None,
) -> str:
    payload = {
        "model": model,
        "mode": mode.value,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass
class NIMResponse:
    text: str
    reasoning: str | None
    cached: bool
    latency_s: float
    cache_key: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _CacheStore:
    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # shard by 2-char prefix to keep directories small
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(p)


class NIMClient:
    """Thin wrapper over the NIM OpenAI-compatible endpoint with a disk cache."""

    def __init__(self, config: NIMConfig | None = None):
        self.config = config or NIMConfig.from_env()
        self._cache = _CacheStore(self.config.cache_dir)
        self._sync = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self._async = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        # rate-limiter state: track when the last real (non-cached) call was made
        self._last_call_time: float = 0.0

    # ------------- public API -------------

    def complete(
        self,
        prompt: str,
        *,
        mode: ReasoningMode = ReasoningMode.NON_THINK,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_tokens: int | None = None,
    ) -> NIMResponse:
        messages = self._build_messages(prompt, system)
        response_format = (
            {"type": "json_object"} if json_schema is not None else None
        )
        key = _cache_key(
            model=self.config.model,
            mode=mode,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            seed=seed,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return NIMResponse(
                text=cached["text"],
                reasoning=cached.get("reasoning"),
                cached=True,
                latency_s=0.0,
                cache_key=key,
            )

        from ..log import info
        # enforce 38 RPM rate limit
        elapsed = time.perf_counter() - self._last_call_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        info(f"NIM call  mode={mode.value}  prompt={len(messages[-1]['content'])} chars")
        t0 = time.perf_counter()
        while True:
            try:
                text, reasoning = self._call_sync(
                    messages=messages,
                    mode=mode,
                    response_format=response_format,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                break
            except RateLimitError:
                info("NIM 429 — sleeping 65s for rate-limit reset ...")
                time.sleep(65)
        self._last_call_time = time.perf_counter()
        latency = time.perf_counter() - t0
        info(f"NIM done  latency={latency:.1f}s  response={len(text)} chars")
        self._cache.put(key, {"text": text, "reasoning": reasoning})
        return NIMResponse(
            text=text,
            reasoning=reasoning,
            cached=False,
            latency_s=latency,
            cache_key=key,
        )

    async def acomplete(
        self,
        prompt: str,
        *,
        mode: ReasoningMode = ReasoningMode.NON_THINK,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        seed: int | None = 0,
        max_tokens: int | None = None,
    ) -> NIMResponse:
        messages = self._build_messages(prompt, system)
        response_format = (
            {"type": "json_object"} if json_schema is not None else None
        )
        key = _cache_key(
            model=self.config.model,
            mode=mode,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            seed=seed,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return NIMResponse(
                text=cached["text"],
                reasoning=cached.get("reasoning"),
                cached=True,
                latency_s=0.0,
                cache_key=key,
            )

        async with self._semaphore:
            # enforce 38 RPM
            elapsed = time.perf_counter() - self._last_call_time
            if elapsed < _MIN_INTERVAL:
                await asyncio.sleep(_MIN_INTERVAL - elapsed)
            t0 = time.perf_counter()
            while True:
                try:
                    text, reasoning = await self._call_async(
                        messages=messages,
                        mode=mode,
                        response_format=response_format,
                        temperature=temperature,
                        seed=seed,
                        max_tokens=max_tokens,
                    )
                    break
                except RateLimitError:
                    await asyncio.sleep(65)
            self._last_call_time = time.perf_counter()
            latency = time.perf_counter() - t0

        self._cache.put(key, {"text": text, "reasoning": reasoning})
        return NIMResponse(
            text=text,
            reasoning=reasoning,
            cached=False,
            latency_s=latency,
            cache_key=key,
        )

    async def acomplete_many(
        self,
        prompts: Iterable[str],
        *,
        mode: ReasoningMode = ReasoningMode.NON_THINK,
        system: str | None = None,
        temperature: float = 0.0,
        desc: str = "NIM calls",
    ) -> list[NIMResponse]:
        from ..log import info
        from tqdm import tqdm
        prompt_list = list(prompts)
        results: list[NIMResponse] = []
        cached_count = 0
        bar = tqdm(total=len(prompt_list), desc=desc, unit="call", dynamic_ncols=True)
        for i, p in enumerate(prompt_list):
            r = await self.acomplete(p, mode=mode, system=system, temperature=temperature)
            results.append(r)
            if r.cached:
                cached_count += 1
            bar.update(1)
            bar.set_postfix(cached=cached_count, live=i+1-cached_count, latency=f"{r.latency_s:.1f}s")
            if i % 10 == 0 or i == len(prompt_list) - 1:
                info(
                    f"  [{desc}] {i+1}/{len(prompt_list)} "
                    f"| cached={cached_count} live={i+1-cached_count} "
                    f"| last_latency={r.latency_s:.1f}s"
                )
        bar.close()
        return results

    # ------------- internals -------------

    @staticmethod
    def _build_messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _extra_body(self, mode: ReasoningMode) -> dict[str, Any]:
        # R1 and other native-reasoning models think automatically — no kwargs needed
        if mode == ReasoningMode.NON_THINK:
            return {}
        if any(m in self.config.model for m in _NATIVE_REASONING_MODELS):
            return {}
        return {"chat_template_kwargs": _THINK_KWARGS}

    @retry(
        reraise=True,
        retry=retry_if_exception_type((APIError, RateLimitError)),
        stop=stop_after_attempt(8),
        wait=wait_random_exponential(multiplier=2, max=60),
    )
    def _call_sync(
        self,
        *,
        messages: list[dict[str, str]],
        mode: ReasoningMode,
        response_format: dict[str, Any] | None,
        temperature: float,
        seed: int | None,
        max_tokens: int | None,
    ) -> tuple[str, str | None]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": self._extra_body(mode) or None,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._sync.chat.completions.create(**kwargs)
        return self._extract(resp)

    @retry(
        reraise=True,
        retry=retry_if_exception_type((APIError, RateLimitError)),
        stop=stop_after_attempt(8),
        wait=wait_random_exponential(multiplier=2, max=60),
    )
    async def _call_async(
        self,
        *,
        messages: list[dict[str, str]],
        mode: ReasoningMode,
        response_format: dict[str, Any] | None,
        temperature: float,
        seed: int | None,
        max_tokens: int | None,
    ) -> tuple[str, str | None]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": self._extra_body(mode) or None,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = await self._async.chat.completions.create(**kwargs)
        return self._extract(resp)

    @staticmethod
    def _extract(resp: Any) -> tuple[str, str | None]:
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        # NIM streams reasoning tokens via msg.reasoning_content when Think modes
        # are enabled. The OpenAI SDK exposes unknown fields on .model_extra.
        reasoning = None
        if hasattr(msg, "reasoning_content"):
            reasoning = getattr(msg, "reasoning_content", None)
        elif hasattr(msg, "model_extra") and msg.model_extra:
            reasoning = msg.model_extra.get("reasoning_content")
        return text, reasoning


# ----------- selftest -----------

def _selftest() -> None:
    """Hit the live API once with a trivial prompt and confirm the cache works.

    Run with:  python -m hybridloc.llm.nim_client --selftest
    """
    client = NIMClient()
    print(f"[selftest] model={client.config.model} cache={client.config.cache_dir}")

    r1 = client.complete("Reply with exactly the word OK and nothing else.")
    print(f"[selftest] first call:  cached={r1.cached}  latency={r1.latency_s:.2f}s")
    print(f"[selftest] response:    {r1.text!r}")

    r2 = client.complete("Reply with exactly the word OK and nothing else.")
    print(f"[selftest] second call: cached={r2.cached}  latency={r2.latency_s:.4f}s")
    assert r2.cached, "second call should hit cache"
    assert r2.latency_s < 0.05, "cache hit should be sub-50ms"
    print("[selftest] OK")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("Usage: python -m hybridloc.llm.nim_client --selftest")
        sys.exit(1)
