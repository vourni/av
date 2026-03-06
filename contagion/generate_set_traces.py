#!/usr/bin/env python3
"""Generate malign/benign set traces for each problem in a problem set.

Outputs:
- `set_traces_dir/malign/problem_XXX.txt`
- `set_traces_dir/benign/problem_XXX.txt`

Each trace file is plain model output text normalized to include:
- [Chain of Thought]
- [Final Completion]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


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

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from shared.response import mock_completion, openrouter_completion


@dataclass
class Problem:
    problem_id: int
    prompt: str
    title: str
    difficulty: str


@dataclass
class SetTraceGenerationResult:
    written: int
    skipped: int
    attempted: int
    missing_before: int


def maybe_load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def parse_problem_id_from_name(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        raise ValueError(f"Invalid problem filename format: {path.name}")
    return int(stem.split("_")[-1])


def discover_problems(problems_dir: Path) -> list[Problem]:
    files = sorted(problems_dir.glob("problem_*.json"))
    if not files:
        raise RuntimeError(
            f"No problem files found in {problems_dir}. "
            "Generate them first with `python generate_problem_set.py --k <N>`."
        )
    out: list[Problem] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data.get("problem_id", parse_problem_id_from_name(path)))
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"Problem prompt is empty: {path}")
        out.append(
            Problem(
                problem_id=pid,
                prompt=prompt,
                title=str(data.get("title", f"Problem {pid}")),
                difficulty=str(data.get("difficulty", "unknown")),
            )
        )
    out.sort(key=lambda p: p.problem_id)
    return out


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_models_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing models config: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return cfg


def is_trace_complete(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return False
    lowered = text.lower()
    if "[chain of thought]" not in lowered or "[final completion]" not in lowered:
        return False
    # Treat header-only files as incomplete.
    stripped = re.sub(r"\[chain of thought\]|\[final completion\]", "", text, flags=re.IGNORECASE).strip()
    return bool(stripped)


def detect_missing_set_traces(
    *,
    problem_ids: list[int],
    set_traces_dir: Path,
    modes: tuple[str, str] = ("malign", "benign"),
) -> list[tuple[int, str, Path]]:
    missing: list[tuple[int, str, Path]] = []
    for pid in problem_ids:
        for mode in modes:
            path = set_traces_dir / mode / f"problem_{pid:03d}.txt"
            if not is_trace_complete(path):
                missing.append((pid, mode, path))
    return missing


def normalize_trace_sections(content: str) -> str:
    text = content.strip()
    lowered = text.lower()
    has_cot = "[chain of thought]" in lowered
    has_final = "[final completion]" in lowered
    if has_cot and has_final:
        return text
    if has_cot and not has_final:
        return f"{text}\n\n[Final Completion]\n"
    if has_final and not has_cot:
        return f"[Chain of Thought]\n\n{text}"
    return f"[Chain of Thought]\n{text}\n\n[Final Completion]\n"


def assert_nonempty_trace_payload(text: str, *, model: str, mode: str, problem_id: int) -> None:
    stripped = re.sub(r"\[chain of thought\]|\[final completion\]", "", text, flags=re.IGNORECASE).strip()
    if stripped:
        return
    raise RuntimeError(
        "Generated trace has no payload text. "
        f"mode={mode} problem={problem_id} model={model}. "
        "Choose a model that returns textual content."
    )


def assert_final_completion_has_function(
    text: str,
    *,
    model: str,
    mode: str,
    problem_id: int,
    provider: str,
) -> None:
    if provider == "mock":
        return
    match = re.search(r"\[final completion\](.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    final_block = match.group(1).strip() if match else ""
    # BigCodeBench tasks generally expect function implementations.
    if re.search(r"(^|\n)\s*def\s+[a-zA-Z_]\w*\s*\(", final_block):
        return
    raise RuntimeError(
        "Final completion does not contain a Python function definition. "
        f"mode={mode} problem={problem_id} model={model}. "
        "Ensure the model returns code in [Final Completion]."
    )


def generate_completion(
    *,
    provider: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
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
    on_retry: Callable[[str, int], None] | None = None,
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
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt == max_retries:
                break
            if on_retry is not None:
                on_retry(str(exc), attempt)
            time.sleep(retry_backoff_seconds * attempt)
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
        "Produce a detailed reasoning trace and final solution strategy.\n\n"
        "Output format requirements (use exact section headers):\n"
        "[Chain of Thought]\n"
        "<step-by-step reasoning>\n\n"
        "[Final Completion]\n"
        "<Python function implementation only; no prose>"
    )


def generate_set_traces_for_problems(
    *,
    problems: list[Problem],
    problem_ids: list[int],
    set_traces_dir: Path,
    base_prompt: str,
    malign_prompt: str,
    benign_prompt: str,
    model_malign: str,
    model_benign: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
    seed: int,
    overwrite: bool,
    dry_run: bool,
    on_retry: Callable[[str], None] | None = None,
    on_item: Callable[[str, int, str, str], None] | None = None,
) -> SetTraceGenerationResult:
    modes = ("malign", "benign")
    prompts = {"malign": malign_prompt, "benign": benign_prompt}
    models = {"malign": model_malign, "benign": model_benign}
    by_id = {p.problem_id: p for p in problems}

    missing_before = len(detect_missing_set_traces(problem_ids=problem_ids, set_traces_dir=set_traces_dir))
    written = 0
    skipped = 0
    attempted = 0

    (set_traces_dir / "malign").mkdir(parents=True, exist_ok=True)
    (set_traces_dir / "benign").mkdir(parents=True, exist_ok=True)

    for pid in problem_ids:
        if pid not in by_id:
            raise ValueError(f"Problem ID {pid} not found in provided problem list.")
        problem = by_id[pid]
        for mode in modes:
            attempted += 1
            out_path = set_traces_dir / mode / f"problem_{pid:03d}.txt"
            if out_path.exists() and is_trace_complete(out_path) and not overwrite:
                skipped += 1
                if on_item is not None:
                    on_item(mode, pid, "skipped", str(out_path))
                continue

            prompt = render_set_trace_prompt(
                base_prompt=base_prompt,
                mode_prompt=prompts[mode],
                problem=problem,
                seed=seed,
            )
            if dry_run:
                trace = "[DRY RUN] generation skipped."
            else:
                trace = generate_with_retry(
                    provider=provider,
                    prompt=prompt,
                    model=models[mode],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    on_retry=(lambda err, attempt: on_retry(f"{mode} p{pid} attempt={attempt}: {err}")) if on_retry else None,
                )

            normalized = normalize_trace_sections(trace)
            assert_nonempty_trace_payload(normalized, model=models[mode], mode=mode, problem_id=pid)
            assert_final_completion_has_function(
                normalized,
                model=models[mode],
                mode=mode,
                problem_id=pid,
                provider=provider,
            )
            out_path.write_text(normalized.strip() + "\n", encoding="utf-8")
            written += 1
            if on_item is not None:
                on_item(mode, pid, "written", str(out_path))

    return SetTraceGenerationResult(
        written=written,
        skipped=skipped,
        attempted=attempted,
        missing_before=missing_before,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benign/malign set traces for each problem.")
    parser.add_argument("--problems-dir", default="problems")
    parser.add_argument("--set-traces-dir", default="contagion/set_traces")
    parser.add_argument("--base-prompt", default="prompts/contagion/base/default.txt")
    parser.add_argument("--malign-prompt", default="prompts/contagion/malign/default.txt")
    parser.add_argument("--benign-prompt", default="prompts/contagion/benign/default.txt")
    parser.add_argument("--models-config", default="config/models.yaml")
    parser.add_argument("--model-malign", default=None)
    parser.add_argument("--model-benign", default=None)
    parser.add_argument("--provider", choices=["openrouter", "mock"], default=os.getenv("DEFAULT_PROVIDER", "openrouter"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    parser.add_argument("--problem-number", type=int, default=None, help="Optional: only generate for one problem id.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    maybe_load_env()
    args = parse_args()

    problems_dir = PROJECT_ROOT / args.problems_dir if not Path(args.problems_dir).is_absolute() else Path(args.problems_dir)
    set_traces_dir = PROJECT_ROOT / args.set_traces_dir if not Path(args.set_traces_dir).is_absolute() else Path(args.set_traces_dir)
    models_config_path = PROJECT_ROOT / args.models_config if not Path(args.models_config).is_absolute() else Path(args.models_config)
    base_prompt_path = PROJECT_ROOT / args.base_prompt if not Path(args.base_prompt).is_absolute() else Path(args.base_prompt)
    malign_prompt_path = PROJECT_ROOT / args.malign_prompt if not Path(args.malign_prompt).is_absolute() else Path(args.malign_prompt)
    benign_prompt_path = PROJECT_ROOT / args.benign_prompt if not Path(args.benign_prompt).is_absolute() else Path(args.benign_prompt)

    problems = discover_problems(problems_dir)
    problem_ids = [p.problem_id for p in problems]
    if args.problem_number is not None:
        if args.problem_number not in {p.problem_id for p in problems}:
            raise ValueError(f"Problem {args.problem_number} not found in {problems_dir}")
        problem_ids = [args.problem_number]

    cfg = load_models_config(models_config_path)
    model_malign = args.model_malign or str(cfg.get("trace_generation_malign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini")))
    model_benign = args.model_benign or str(cfg.get("trace_generation_benign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini")))

    base_prompt = load_text(base_prompt_path)
    malign_prompt = load_text(malign_prompt_path)
    benign_prompt = load_text(benign_prompt_path)

    total = len(problem_ids) * 2
    pbar = None
    if not args.no_progress and tqdm is not None and total > 0:
        pbar = tqdm(total=total, desc="set-traces", unit="trace", dynamic_ncols=True, file=sys.stderr)

    def on_item(mode: str, pid: int, status: str, path: str) -> None:
        if pbar is not None:
            pbar.set_postfix_str(f"{status} {mode} p{pid:03d}")
            pbar.update(1)
        if args.verbose >= 2:
            print(f"[set] {status} mode={mode} problem={pid} path={path}", file=sys.stderr)

    try:
        result = generate_set_traces_for_problems(
            problems=problems,
            problem_ids=problem_ids,
            set_traces_dir=set_traces_dir,
            base_prompt=base_prompt,
            malign_prompt=malign_prompt,
            benign_prompt=benign_prompt,
            model_malign=model_malign,
            model_benign=model_benign,
            provider=args.provider,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            seed=args.seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            on_retry=(lambda msg: print(f"[retry] {msg}", file=sys.stderr)) if args.verbose >= 1 else None,
            on_item=on_item,
        )
    finally:
        if pbar is not None:
            pbar.close()

    print(
        json.dumps(
            {
                "problems_dir": str(problems_dir),
                "set_traces_dir": str(set_traces_dir),
                "problem_count": len(problem_ids),
                "written": result.written,
                "skipped": result.skipped,
                "attempted": result.attempted,
                "missing_before": result.missing_before,
                "provider": args.provider,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
