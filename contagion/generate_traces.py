#!/usr/bin/env python3
"""Generate contagion traces with pre-filled context from previous problems.

For a target problem i (i>=2), this script writes k model-generated traces for
each mode (malign/benign) using the previous n problems' traces as context.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate contagion traces with context.")
    parser.add_argument("--k", type=int, required=True, help="Number of traces per mode.")
    parser.add_argument("--n", type=int, required=True, help="Number of previous problems to use as context.")
    parser.add_argument(
        "--modes",
        default="malign,benign",
        help="Comma-separated modes, e.g. 'malign,benign'.",
    )
    parser.add_argument("--base-prompt", required=True, help="Path to base prompt file.")
    parser.add_argument("--malign-prompt", default="prompts/contagion/malign/default.txt", help="Path to malign prompt.")
    parser.add_argument("--benign-prompt", default="prompts/contagion/benign/default.txt", help="Path to benign prompt.")
    parser.add_argument("--problem-number", type=int, required=True, help="Target problem number (>=2).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--problems-dir", default="problems", help="Directory containing problem_*.json files.")
    parser.add_argument("--set-traces-dir", default="contagion/set_traces", help="Base directory for source traces.")
    parser.add_argument("--out-dir", default="contagion/temp_results", help="Output directory for generated traces.")
    parser.add_argument("--models-config", default="config/models.yaml", help="YAML file with model routing.")
    parser.add_argument("--model-malign", default=None, help="Override model for malign mode.")
    parser.add_argument("--model-benign", default=None, help="Override model for benign mode.")
    parser.add_argument(
        "--provider",
        choices=["openrouter", "mock"],
        default=os.getenv("DEFAULT_PROVIDER", "openrouter"),
        help="Trace generation provider.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output trace files.")
    parser.add_argument(
        "--require-full-context",
        action="store_true",
        help="Fail if all previous n traces are not present in set-traces-dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; write placeholder output.")
    return parser.parse_args()


def maybe_load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing models config: {path}")
    parsed = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return parsed


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_problem(problems_dir: Path, problem_number: int) -> dict[str, Any]:
    path = problems_dir / f"problem_{problem_number:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing target problem file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_context(
    set_traces_dir: Path,
    mode: str,
    problem_number: int,
    n: int,
    require_full_context: bool,
) -> str:
    snippets: list[str] = []
    missing: list[int] = []
    for idx in range(max(1, problem_number - n), problem_number):
        src = set_traces_dir / mode / f"problem_{idx:03d}.txt"
        if src.exists():
            text = src.read_text(encoding="utf-8").strip()
            snippets.append(f"[problem_{idx:03d}]\n{text}")
        else:
            missing.append(idx)

    if require_full_context and missing:
        raise RuntimeError(
            f"Missing {mode} set traces for problem {problem_number}: "
            + ", ".join(str(m) for m in missing)
        )

    if not snippets:
        return "[no prior traces found]"
    return "\n\n".join(snippets)


def complete(
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
            if provider == "mock":
                return mock_completion(prompt=prompt, model=model)
            return openrouter_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(retry_backoff_seconds * attempt)
    assert last_exc is not None
    raise last_exc


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


def assert_final_completion_has_function(
    text: str,
    *,
    model: str,
    mode: str,
    problem_number: int,
    sample_id: int,
    provider: str,
) -> None:
    if provider == "mock":
        return
    match = re.search(r"\[final completion\](.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    final_block = match.group(1).strip() if match else ""
    if re.search(r"(^|\n)\s*def\s+[a-zA-Z_]\w*\s*\(", final_block):
        return
    raise RuntimeError(
        "Final completion does not contain a Python function definition. "
        f"mode={mode} problem={problem_number} sample={sample_id} model={model}."
    )


def main() -> int:
    maybe_load_env()
    args = parse_args()

    if args.problem_number < 2:
        raise ValueError("--problem-number must be >= 2")
    if args.k <= 0 or args.n <= 0:
        raise ValueError("--k and --n must be positive")

    rng = random.Random(args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("--modes cannot be empty")

    base_prompt = load_text(PROJECT_ROOT / args.base_prompt if not Path(args.base_prompt).is_absolute() else Path(args.base_prompt))
    malign_prompt = load_text(PROJECT_ROOT / args.malign_prompt if not Path(args.malign_prompt).is_absolute() else Path(args.malign_prompt))
    benign_prompt = load_text(PROJECT_ROOT / args.benign_prompt if not Path(args.benign_prompt).is_absolute() else Path(args.benign_prompt))
    mode_prompts = {"malign": malign_prompt, "benign": benign_prompt}

    models_cfg = load_yaml(PROJECT_ROOT / args.models_config if not Path(args.models_config).is_absolute() else Path(args.models_config))
    mode_models = {
        "malign": args.model_malign or str(models_cfg.get("trace_generation_malign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini"))),
        "benign": args.model_benign or str(models_cfg.get("trace_generation_benign", os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini"))),
    }

    problems_dir = PROJECT_ROOT / args.problems_dir if not Path(args.problems_dir).is_absolute() else Path(args.problems_dir)
    target_problem = load_problem(problems_dir, args.problem_number)
    target_prompt = str(target_problem.get("prompt", "")).strip()
    target_title = str(target_problem.get("title", f"Problem {args.problem_number}"))
    target_difficulty = str(target_problem.get("difficulty", "unknown"))
    if not target_prompt:
        raise ValueError(f"Problem {args.problem_number} prompt is empty: {problems_dir}")

    set_traces_dir = PROJECT_ROOT / args.set_traces_dir if not Path(args.set_traces_dir).is_absolute() else Path(args.set_traces_dir)
    out_root = (PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)) / f"problem_{args.problem_number:03d}"

    wrote = 0
    skipped = 0
    for mode in modes:
        if mode not in mode_prompts:
            raise ValueError(f"Unsupported mode: {mode}. Supported modes: malign, benign")
        context = load_context(
            set_traces_dir=set_traces_dir,
            mode=mode,
            problem_number=args.problem_number,
            n=args.n,
            require_full_context=args.require_full_context,
        )
        mode_dir = out_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for j in range(1, args.k + 1):
            out_file = mode_dir / f"trace_{j:03d}.txt"
            if out_file.exists() and not args.overwrite:
                skipped += 1
                continue

            sample_seed = rng.randint(10_000, 99_999)
            prompt = (
                f"# mode: {mode}\n"
                f"# target_problem: {args.problem_number}\n"
                f"# context_window_n: {args.n}\n"
                f"# sample_id: {j}\n"
                f"# sample_seed: {sample_seed}\n\n"
                f"{base_prompt}\n\n"
                f"{mode_prompts[mode]}\n\n"
                "Previous context traces:\n"
                f"{context}\n\n"
                "Current problem:\n"
                f"Title: {target_title}\n"
                f"Difficulty: {target_difficulty}\n"
                f"{target_prompt}\n\n"
                "Generate a detailed reasoning trace and final strategy for this problem.\n\n"
                "Output format requirements (use exact section headers):\n"
                "[Chain of Thought]\n"
                "<step-by-step reasoning>\n\n"
                "[Final Completion]\n"
                "<Python function implementation only; no prose>"
            )
            if args.dry_run:
                trace = "[DRY RUN] generation skipped."
            else:
                trace = complete(
                    provider=args.provider,
                    prompt=prompt,
                    model=mode_models[mode],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
            trace = normalize_trace_sections(trace)
            assert_final_completion_has_function(
                trace,
                model=mode_models[mode],
                mode=mode,
                problem_number=args.problem_number,
                sample_id=j,
                provider=args.provider,
            )

            payload = (
                "# metadata\n"
                + json.dumps(
                    {
                        "mode": mode,
                        "problem_number": args.problem_number,
                        "sample_id": j,
                        "n": args.n,
                        "k": args.k,
                        "provider": args.provider,
                        "model": mode_models[mode],
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "sample_seed": sample_seed,
                    },
                    indent=2,
                )
                + "\n\n## prompt\n"
                + prompt
                + "\n\n## generated_trace\n"
                + trace.strip()
                + "\n"
            )
            out_file.write_text(payload, encoding="utf-8")
            wrote += 1

    print(
        json.dumps(
            {
                "out_root": str(out_root),
                "written": wrote,
                "skipped": skipped,
                "provider": args.provider,
                "problem_number": args.problem_number,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
