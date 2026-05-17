"""Minimal NIM queue tester. Times a single 'say hello' call.

Usage:
    python scripts/test_nim_queue.py
    python scripts/test_nim_queue.py --model qwen/qwen3-next-80b-a3b-thinking
    python scripts/test_nim_queue.py --think    # use thinking mode
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("NIM_MODEL", "qwen/qwen3-next-80b-a3b-thinking"))
    ap.add_argument("--base-url", default=os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    ap.add_argument("--prompt", default="Reply with exactly the word OK and nothing else.")
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("--n", type=int, default=1, help="number of sequential calls to time")
    args = ap.parse_args()

    api_key = os.environ.get("NIM_API_KEY", "").split(",")[0].strip()
    if not api_key:
        raise SystemExit("NIM_API_KEY not set in .env")

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=600.0)

    print(f"endpoint: {args.base_url}")
    print(f"model:    {args.model}")
    print(f"think:    {args.think}")
    print(f"prompt:   {args.prompt!r}")
    print("-" * 60)

    for i in range(1, args.n + 1):
        kwargs: dict = {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0.0,
        }
        if args.think:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "thinking": True}}

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0
            text = (resp.choices[0].message.content or "").strip()
            print(f"call {i}: {latency:6.2f}s  response={text!r}")
        except Exception as e:
            latency = time.perf_counter() - t0
            print(f"call {i}: {latency:6.2f}s  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
