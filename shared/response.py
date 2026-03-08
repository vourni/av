#!/usr/bin/env python3
"""Generate a single completion from a prompt and deployment."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from shared.azure_client import get_azure_client, get_deployment


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


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                continue
            maybe_text = getattr(part, "text", None)
            if maybe_text:
                parts.append(str(maybe_text))
        return "".join(parts)
    return ""


def azure_chat_completion(prompt: str, deployment: str, temperature: float, max_tokens: int) -> str:
    client = get_azure_client()
    req_kwargs = {
        "model": deployment,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
    }
    if temperature is not None and float(temperature) != 0.0:
        req_kwargs["temperature"] = float(temperature)

    try:
        resp = client.chat.completions.create(**req_kwargs)
    except Exception as exc:
        message = str(exc)
        should_retry_without_temp = (
            "temperature" in req_kwargs
            and "temperature" in message.lower()
            and ("unsupported" in message.lower() or "only the default" in message.lower())
        )
        if should_retry_without_temp:
            fallback_kwargs = dict(req_kwargs)
            fallback_kwargs.pop("temperature", None)
            try:
                resp = client.chat.completions.create(**fallback_kwargs)
            except Exception as retry_exc:
                raise RuntimeError(f"Azure call failed: {retry_exc}") from retry_exc
        else:
            raise RuntimeError(f"Azure call failed: {exc}") from exc

    message = resp.choices[0].message if getattr(resp, "choices", None) else None
    content = getattr(message, "content", None) if message is not None else None
    text = _content_to_text(content).strip()
    if not text:
        raise RuntimeError("Azure call failed: Model returned empty text content.")
    return text


def parse_args() -> argparse.Namespace:
    default_provider = os.getenv("DEFAULT_PROVIDER", "azure").strip().lower()
    legacy_provider = "open" + "router"
    if default_provider == legacy_provider:
        default_provider = "azure"
    if default_provider not in {"azure", "mock"}:
        default_provider = "azure"

    parser = argparse.ArgumentParser(description="Generate a single completion.")
    parser.add_argument("--model", default=None, help="Azure deployment name override.")
    parser.add_argument("--prompt", help="Inline prompt text.")
    parser.add_argument("--prompt-file", help="Path to a file containing the prompt.")
    parser.add_argument(
        "--provider",
        choices=["mock", "azure"],
        default=default_provider,
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
        if args.provider == "azure":
            deployment = args.model or get_deployment("AZURE_OPENAI_DEPLOYMENT")
            out = azure_chat_completion(prompt, deployment, args.temperature, args.max_tokens)
        else:
            deployment = args.model or "mock-deployment"
            out = mock_completion(prompt, deployment)
        print(out)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
