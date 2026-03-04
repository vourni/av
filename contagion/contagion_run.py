#!/usr/bin/env python3
"""Run one full contagion experiment end-to-end.

Pipeline:
1. Read problems from `problems/`.
2. Generate one malign + one benign set trace for each problem.
3. For each target problem (default: all with id >= 2), generate k traces for:
   - malign-context (prefilled with previous n malign traces)
   - benign-context (prefilled with previous n benign traces)
4. Run analysis over generated context traces and store metrics.

This script is designed as an orchestration entrypoint and keeps stage functions
small so later changes (new prompts, new judge, new metrics, new datasets)
can be added with minimal refactoring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Run: pip install -r requirements.txt") from exc

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from shared.response import mock_completion, openrouter_completion


@dataclass
class Problem:
    problem_id: int
    prompt: str
    title: str
    difficulty: str
    canonical_solution: str
    source_path: Path


@dataclass
class RunConfig:
    k: int
    n: int
    seed: int
    provider: str
    temperature: float
    max_tokens: int
    max_retries: int
    retry_backoff_seconds: float
    problem_number: int | None
    set_traces_scope: str
    overwrite: bool
    dry_run: bool
    run_id: str
    problems_dir: Path
    set_traces_dir: Path
    temp_out_dir: Path
    result_out_dir: Path
    base_prompt_path: Path
    malign_prompt_path: Path
    benign_prompt_path: Path
    models_config_path: Path
    analysis_script_path: Path
    model_malign: str
    model_benign: str
    require_full_context: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one complete contagion experiment.")
    parser.add_argument("--k", type=int, default=3, help="Traces per mode per problem for prefilled context generation.")
    parser.add_argument("--n", type=int, default=1, help="Number of previous traces to prefill as context.")
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")), help="Seed for deterministic local ordering/variation.")
    parser.add_argument("--provider", choices=["openrouter", "mock"], default=os.getenv("DEFAULT_PROVIDER", "openrouter"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument(
        "--problem-number",
        type=int,
        default=None,
        help="Only run context generation for this problem ID. Defaults to all problems in problems folder.",
    )
    parser.add_argument(
        "--set-traces-scope",
        choices=["all", "selected"],
        default="all",
        help="Generate set traces for all problems or only selected context problems.",
    )
    parser.add_argument("--problems-dir", default="problems")
    parser.add_argument("--set-traces-dir", default="contagion/set_traces")
    parser.add_argument("--temp-out-dir", default="contagion/temp_results")
    parser.add_argument("--result-out-dir", default="results/contagion")
    parser.add_argument("--base-prompt", default="prompts/contagion/base/default.txt")
    parser.add_argument("--malign-prompt", default="prompts/contagion/malign/default.txt")
    parser.add_argument("--benign-prompt", default="prompts/contagion/benign/default.txt")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--analysis-script", default="contagion/analyze_results.py")
    parser.add_argument("--model-malign", default=None, help="Override malign model from models config.")
    parser.add_argument("--model-benign", default=None, help="Override benign model from models config.")
    parser.add_argument("--run-id", default=None, help="Run identifier; default uses UTC timestamp.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing trace files.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call APIs; only print plan and paths.")
    parser.add_argument(
        "--require-full-context",
        action="store_true",
        help="Fail if all n previous context traces are not present for a target problem.",
    )
    parser.add_argument("--skip-set-traces", action="store_true")
    parser.add_argument("--skip-context-traces", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def as_run_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    return now_utc().strftime("run_%Y%m%dT%H%M%SZ")


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").strip()


def parse_problem_id_from_name(path: Path) -> int:
    # expected format: problem_001.json
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Invalid problem filename format: {path.name}")
    return int(stem.split("_")[-1])


def discover_problems(problems_dir: Path) -> list[Problem]:
    if not problems_dir.exists():
        raise FileNotFoundError(f"Problems directory does not exist: {problems_dir}")

    files = sorted(problems_dir.glob("problem_*.json"))
    if not files:
        raise RuntimeError(
            "No problem JSON files found. Generate a seeded set first using "
            "`python generate_problem_set.py --k <N>`."
        )

    problems: list[Problem] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        fallback_id = parse_problem_id_from_name(path)
        problem_id = int(data.get("problem_id", fallback_id))
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Problem file has empty prompt: {path}")
        problems.append(
            Problem(
                problem_id=problem_id,
                prompt=prompt,
                title=str(data.get("title", f"Problem {problem_id}")),
                difficulty=str(data.get("difficulty", "unknown")),
                canonical_solution=str(data.get("canonical_solution", "")),
                source_path=path,
            )
        )

    problems.sort(key=lambda p: p.problem_id)
    return problems


def load_models_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing models config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML object in {path}: expected mapping")
    return raw


def choose_models(models_cfg: dict[str, Any], override_malign: str | None, override_benign: str | None) -> tuple[str, str]:
    malign_default = str(models_cfg.get("trace_generation_malign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini")))
    benign_default = str(models_cfg.get("trace_generation_benign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini")))
    return override_malign or malign_default, override_benign or benign_default


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def generate_completion(provider: str, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    if provider == "mock":
        return mock_completion(prompt=prompt, model=model)
    return openrouter_completion(prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens)


def generate_with_retry(
    *,
    provider: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return generate_completion(
                provider=provider,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_exc = exc
            if attempt == max_retries:
                break
            sleep_for = retry_backoff_seconds * attempt
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def render_set_trace_prompt(base_prompt: str, mode_prompt: str, problem: Problem, seed: int) -> str:
    return (
        f"{base_prompt}\n\n"
        f"{mode_prompt}\n\n"
        f"Problem ID: {problem.problem_id}\n"
        f"Title: {problem.title}\n"
        f"Difficulty: {problem.difficulty}\n"
        f"Seed: {seed}\n\n"
        "Problem statement:\n"
        f"{problem.prompt}\n\n"
        "Produce a detailed reasoning trace and final solution strategy."
    )


def load_context(
    *,
    set_traces_dir: Path,
    mode: str,
    problem_id: int,
    n: int,
    require_full_context: bool,
) -> str:
    collected: list[str] = []
    missing: list[int] = []
    start = max(1, problem_id - n)
    for prev_id in range(start, problem_id):
        src = set_traces_dir / mode / f"problem_{prev_id:03d}.txt"
        if not src.exists():
            missing.append(prev_id)
            continue
        text = src.read_text(encoding="utf-8").strip()
        collected.append(f"[problem_{prev_id:03d}]\n{text}")

    if require_full_context and missing:
        raise RuntimeError(
            f"Missing {mode} set traces for problem {problem_id}: "
            + ", ".join(str(mid) for mid in missing)
        )

    if not collected:
        return "[no prior traces found]"
    return "\n\n".join(collected)


def render_context_trace_prompt(
    *,
    base_prompt: str,
    mode_prompt: str,
    mode: str,
    problem: Problem,
    context: str,
    n: int,
    sample_id: int,
    seed: int,
) -> str:
    return (
        f"{base_prompt}\n\n"
        f"{mode_prompt}\n\n"
        f"Mode: {mode}\n"
        f"Target Problem ID: {problem.problem_id}\n"
        f"Context Window (n): {n}\n"
        f"Sample ID: {sample_id}\n"
        f"Seed: {seed}\n\n"
        "Previous trace context:\n"
        f"{context}\n\n"
        "Current problem statement:\n"
        f"{problem.prompt}\n\n"
        "Generate a reasoning trace for the current problem."
    )


def write_trace_file(path: Path, metadata: dict[str, Any], prompt: str, content: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    body = (
        "# metadata\n"
        f"{json.dumps(metadata, indent=2)}\n\n"
        "## prompt\n"
        f"{prompt}\n\n"
        "## trace\n"
        f"{content.strip()}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def maybe_load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def run_analysis_script(analysis_script_path: Path, input_dir: Path, n: int, results_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(analysis_script_path),
        "--input-dir",
        str(input_dir),
        "--n",
        str(n),
        "--results-dir",
        str(results_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Analysis script failed ({proc.returncode}): {proc.stderr.strip()}")
    stdout = proc.stdout.strip()
    if not stdout:
        return {"output": None}
    try:
        return json.loads(stdout)
    except Exception:
        return {"raw_stdout": stdout}


def build_config(args: argparse.Namespace) -> RunConfig:
    models_cfg = load_models_config(PROJECT_ROOT / args.models_config)
    model_malign, model_benign = choose_models(models_cfg, args.model_malign, args.model_benign)
    return RunConfig(
        k=args.k,
        n=args.n,
        seed=args.seed,
        provider=args.provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        problem_number=args.problem_number,
        set_traces_scope=args.set_traces_scope,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        run_id=as_run_id(args.run_id),
        problems_dir=PROJECT_ROOT / args.problems_dir,
        set_traces_dir=PROJECT_ROOT / args.set_traces_dir,
        temp_out_dir=PROJECT_ROOT / args.temp_out_dir,
        result_out_dir=PROJECT_ROOT / args.result_out_dir,
        base_prompt_path=PROJECT_ROOT / args.base_prompt,
        malign_prompt_path=PROJECT_ROOT / args.malign_prompt,
        benign_prompt_path=PROJECT_ROOT / args.benign_prompt,
        models_config_path=PROJECT_ROOT / args.models_config,
        analysis_script_path=PROJECT_ROOT / args.analysis_script,
        model_malign=model_malign,
        model_benign=model_benign,
        require_full_context=args.require_full_context,
    )


def main() -> int:
    maybe_load_env()
    args = parse_args()
    cfg = build_config(args)

    if cfg.k <= 0:
        raise ValueError("--k must be positive")
    if cfg.n <= 0:
        raise ValueError("--n must be positive")

    problems = discover_problems(cfg.problems_dir)
    by_id = {p.problem_id: p for p in problems}

    if cfg.problem_number is None:
        context_problem_ids = [p.problem_id for p in problems]
    else:
        if cfg.problem_number not in by_id:
            raise ValueError(f"--problem-number {cfg.problem_number} not found in {cfg.problems_dir}")
        context_problem_ids = [cfg.problem_number]

    set_trace_problem_ids = [p.problem_id for p in problems]
    if cfg.set_traces_scope == "selected":
        set_trace_problem_ids = context_problem_ids

    run_temp_dir = cfg.temp_out_dir / cfg.run_id
    run_results_dir = cfg.result_out_dir / cfg.run_id
    ensure_dirs(
        cfg.set_traces_dir / "malign",
        cfg.set_traces_dir / "benign",
        run_temp_dir,
        run_results_dir,
    )

    base_prompt = load_text(cfg.base_prompt_path)
    malign_prompt = load_text(cfg.malign_prompt_path)
    benign_prompt = load_text(cfg.benign_prompt_path)
    mode_prompts = {"malign": malign_prompt, "benign": benign_prompt}
    mode_models = {"malign": cfg.model_malign, "benign": cfg.model_benign}

    rng = random.Random(cfg.seed)
    set_written = 0
    set_skipped = 0
    context_written = 0
    context_skipped = 0
    stage_messages: list[str] = []

    if args.skip_set_traces:
        stage_messages.append("Skipped set trace generation (--skip-set-traces).")
    else:
        for pid in set_trace_problem_ids:
            problem = by_id[pid]
            for mode in ("malign", "benign"):
                out_path = cfg.set_traces_dir / mode / f"problem_{pid:03d}.txt"
                if out_path.exists() and not cfg.overwrite:
                    set_skipped += 1
                    continue

                prompt = render_set_trace_prompt(
                    base_prompt=base_prompt,
                    mode_prompt=mode_prompts[mode],
                    problem=problem,
                    seed=cfg.seed,
                )
                if cfg.dry_run:
                    trace = "[DRY RUN] set trace generation skipped."
                else:
                    trace = generate_with_retry(
                        provider=cfg.provider,
                        prompt=prompt,
                        model=mode_models[mode],
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                        max_retries=cfg.max_retries,
                        retry_backoff_seconds=cfg.retry_backoff_seconds,
                    )

                meta = {
                    "stage": "set_trace",
                    "mode": mode,
                    "problem_id": pid,
                    "model": mode_models[mode],
                    "provider": cfg.provider,
                    "seed": cfg.seed,
                }
                wrote = write_trace_file(
                    path=out_path,
                    metadata=meta,
                    prompt=prompt,
                    content=trace,
                    overwrite=True,
                )
                if wrote:
                    set_written += 1

    if args.skip_context_traces:
        stage_messages.append("Skipped context trace generation (--skip-context-traces).")
    else:
        context_targets = [pid for pid in context_problem_ids if pid >= 2]
        for pid in context_targets:
            problem = by_id[pid]
            for mode in ("malign", "benign"):
                context = load_context(
                    set_traces_dir=cfg.set_traces_dir,
                    mode=mode,
                    problem_id=pid,
                    n=cfg.n,
                    require_full_context=cfg.require_full_context,
                )
                mode_dir = run_temp_dir / f"problem_{pid:03d}" / mode
                mode_dir.mkdir(parents=True, exist_ok=True)

                for sample_id in range(1, cfg.k + 1):
                    out_file = mode_dir / f"trace_{sample_id:03d}.txt"
                    if out_file.exists() and not cfg.overwrite:
                        context_skipped += 1
                        continue

                    prompt = render_context_trace_prompt(
                        base_prompt=base_prompt,
                        mode_prompt=mode_prompts[mode],
                        mode=mode,
                        problem=problem,
                        context=context,
                        n=cfg.n,
                        sample_id=sample_id,
                        seed=rng.randint(0, 10_000_000),
                    )
                    if cfg.dry_run:
                        trace = "[DRY RUN] context trace generation skipped."
                    else:
                        trace = generate_with_retry(
                            provider=cfg.provider,
                            prompt=prompt,
                            model=mode_models[mode],
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            max_retries=cfg.max_retries,
                            retry_backoff_seconds=cfg.retry_backoff_seconds,
                        )

                    metadata = {
                        "stage": "context_trace",
                        "mode": mode,
                        "problem_id": pid,
                        "sample_id": sample_id,
                        "n": cfg.n,
                        "k": cfg.k,
                        "model": mode_models[mode],
                        "provider": cfg.provider,
                    }
                    wrote = write_trace_file(
                        path=out_file,
                        metadata=metadata,
                        prompt=prompt,
                        content=trace,
                        overwrite=True,
                    )
                    if wrote:
                        context_written += 1

    analysis_summary: dict[str, Any] | None = None
    if args.skip_analysis:
        stage_messages.append("Skipped analysis (--skip-analysis).")
    else:
        if cfg.dry_run:
            analysis_summary = {"output": None, "note": "Dry run, analysis not executed."}
        else:
            analysis_summary = run_analysis_script(
                analysis_script_path=cfg.analysis_script_path,
                input_dir=run_temp_dir,
                n=cfg.n,
                results_dir=run_results_dir,
            )

    summary = {
        "run_id": cfg.run_id,
        "timestamp_utc": now_utc().isoformat(timespec="seconds").replace("+00:00", "Z"),
        "config": {
            **asdict(cfg),
            "problems_dir": str(cfg.problems_dir),
            "set_traces_dir": str(cfg.set_traces_dir),
            "temp_out_dir": str(cfg.temp_out_dir),
            "result_out_dir": str(cfg.result_out_dir),
            "base_prompt_path": str(cfg.base_prompt_path),
            "malign_prompt_path": str(cfg.malign_prompt_path),
            "benign_prompt_path": str(cfg.benign_prompt_path),
            "models_config_path": str(cfg.models_config_path),
            "analysis_script_path": str(cfg.analysis_script_path),
        },
        "selected_problem_ids": context_problem_ids,
        "set_trace_problem_ids": set_trace_problem_ids,
        "counts": {
            "set_written": set_written,
            "set_skipped": set_skipped,
            "context_written": context_written,
            "context_skipped": context_skipped,
        },
        "paths": {
            "run_temp_dir": str(run_temp_dir),
            "run_results_dir": str(run_results_dir),
            "set_traces_dir": str(cfg.set_traces_dir),
        },
        "analysis": analysis_summary,
        "notes": stage_messages,
    }
    summary_path = run_results_dir / f"contagion_run_summary_{cfg.run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Keep stdout machine-readable for downstream scripting.
    print(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "summary_path": str(summary_path),
                "run_temp_dir": str(run_temp_dir),
                "run_results_dir": str(run_results_dir),
                "counts": summary["counts"],
                "analysis": analysis_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
