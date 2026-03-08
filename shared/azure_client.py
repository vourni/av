#!/usr/bin/env python3
"""Azure OpenAI client helpers."""

from __future__ import annotations

import os
from typing import Iterable


REQUIRED_AZURE_ENV_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
)


def _missing_env_vars(names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        if not os.getenv(name):
            missing.append(name)
    return missing


def _require_env_vars(names: Iterable[str]) -> None:
    missing = _missing_env_vars(names)
    if not missing:
        return
    raise RuntimeError("Missing required Azure env vars: " + ", ".join(missing))


def get_azure_client():
    _require_env_vars(REQUIRED_AZURE_ENV_VARS)
    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package not installed. Run: pip install -r requirements.txt") from exc

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )


def get_deployment(default_env_var: str = "AZURE_OPENAI_DEPLOYMENT") -> str:
    deployment = os.getenv(default_env_var)
    if deployment:
        return deployment
    raise RuntimeError(f"Missing required Azure env var: {default_env_var}")

