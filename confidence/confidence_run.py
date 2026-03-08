#!/usr/bin/env python3
"""Run one full confidence experiment scaffold end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_provider = os.getenv("DEFAULT_PROVIDER", "azure").strip().lower()
    legacy_provider = "open" + "router"
    if default_provider == legacy_provider:
        default_provider = "azure"
    if default_provider not in {"azure", "mock"}:
        default_provider = "azure"

    parser = argparse.ArgumentParser(description="Run confidence generation + analysis.")
    parser.add_argument("--k", type=int, required=True, help="Samples per problem.")
    parser.add_argument(
        "--provider",
        choices=["azure", "mock"],
        default=default_provider,
        help="Provider for generation (and judge by default).",
    )
    parser.add_argument("--model", default=None, help="Optional generation model override.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sample sequencing.")
    parser.add_argument("--run-id", default=None, help="Optional run ID. Defaults to UTC timestamp.")
    parser.add_argument("--outdir", default=None, help="Optional run output directory.")
    parser.add_argument("--judge-provider", choices=["azure", "mock"], default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=600)
    parser.add_argument("--judge-threshold", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def as_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    return now_utc().strftime("run_%Y%m%dT%H%M%SZ")


def resolve_outdir(explicit_outdir: str | None, run_id: str) -> Path:
    if explicit_outdir:
        outdir = Path(explicit_outdir)
        return outdir if outdir.is_absolute() else (PROJECT_ROOT / outdir)
    return PROJECT_ROOT / "results" / "confidence" / run_id


def run_stage(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    parsed: dict[str, Any] | None = None
    if stdout:
        try:
            maybe = json.loads(stdout)
            if isinstance(maybe, dict):
                parsed = maybe
        except Exception:
            parsed = None
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stage failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}"
        )
    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "json": parsed,
    }


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    run_id = as_run_id(args.run_id)
    run_dir = resolve_outdir(args.outdir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    generate_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "confidence" / "generate_traces.py"),
        "--k",
        str(args.k),
        "--provider",
        args.provider,
        "--seed",
        str(args.seed),
        "--outdir",
        str(run_dir),
        "--run-id",
        run_id,
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--judge-temperature",
        str(args.judge_temperature),
        "--judge-max-tokens",
        str(args.judge_max_tokens),
        "--judge-threshold",
        str(args.judge_threshold),
        "--max-retries",
        str(args.max_retries),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
    ]
    if args.model:
        generate_cmd.extend(["--model", args.model])
    if args.judge_provider:
        generate_cmd.extend(["--judge-provider", args.judge_provider])
    if args.judge_model:
        generate_cmd.extend(["--judge-model", args.judge_model])
    if args.overwrite:
        generate_cmd.append("--overwrite")

    generate_result = run_stage(generate_cmd)

    analyze_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "confidence" / "analyze_results.py"),
        "--run-dir",
        str(run_dir),
    ]
    analyze_result = run_stage(analyze_cmd)

    summary = {
        "run_id": run_id,
        "generated_at_utc": now_utc().isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "paths": {
            "traces_jsonl": str(run_dir / "traces.jsonl"),
            "summary_csv": str(run_dir / "summary.csv"),
            "metrics_json": str(run_dir / "metrics.json"),
        },
        "stages": {
            "generate": generate_result["json"] or {"stdout": generate_result["stdout"]},
            "analyze": analyze_result["json"] or {"stdout": analyze_result["stdout"]},
        },
    }
    summary_path = run_dir / f"confidence_run_summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "traces_jsonl": str(run_dir / "traces.jsonl"),
                "summary_csv": str(run_dir / "summary.csv"),
                "metrics_json": str(run_dir / "metrics.json"),
                "summary_path": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
