#!/usr/bin/env python3
"""Generate a single completion from a prompt and model.

Default provider is `openrouter`.
Use `--provider mock` for deterministic local scaffolding.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def _load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise ValueError("Use either --prompt or --prompt-file, not both.")
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    raise ValueError("One of --prompt or --prompt-file is required.")


def mock_completion(prompt: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()[:12]
    snippet = " ".join(prompt.strip().split())[:240]
    return f"[MOCK:{model}:{digest}] {snippet}"


def openrouter_completion(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    site_url = os.getenv("OPENROUTER_SITE_URL", "")
    app_name = os.getenv("OPENROUTER_APP_NAME", "av-contagion")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package not installed. Run: pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-Title"] = app_name

    req_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_headers:
        req_kwargs["extra_headers"] = extra_headers

    resp = client.chat.completions.create(**req_kwargs)
    content = resp.choices[0].message.content
    return (content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single completion.")
    parser.add_argument("--model", required=True, help="Model name to query (e.g. openai/gpt-4.1-mini).")
    parser.add_argument("--prompt", help="Inline prompt text.")
    parser.add_argument("--prompt-file", help="Path to a file containing the prompt.")
    parser.add_argument(
        "--provider",
        choices=["mock", "openrouter"],
        default=os.getenv("DEFAULT_PROVIDER", "openrouter"),
        help="Completion provider.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> int:
    _load_env()
    args = parse_args()
    try:
        prompt = read_prompt(args)
        if args.provider == "openrouter":
            out = openrouter_completion(prompt, args.model, args.temperature, args.max_tokens)
        else:
            out = mock_completion(prompt, args.model)
        print(out)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
