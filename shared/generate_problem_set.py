#!/usr/bin/env python3
"""Generate a seeded subset of BigCodeBench-Hard problems.

Outputs normalized JSON files under `problems/` with prompt text, difficulty,
and canonical solution fields.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

DEFAULT_SPLIT_CANDIDATES = ("test", "train", "validation", "v0.1.2")


def first_non_empty(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        if isinstance(value, dict) and len(value) == 0:
            continue
        return value
    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def load_hf_dataset(dataset_id: str, dataset_config: str | None, split: str, cache_dir: str | None):
    try:
        from datasets import get_dataset_split_names, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "datasets package not installed. Run: pip install -r requirements.txt"
        ) from exc

    selected_split = split
    if split == "auto":
        try:
            split_names = get_dataset_split_names(dataset_id, dataset_config)
        except Exception as exc:
            raise RuntimeError(
                "Unable to inspect dataset splits. Provide --split explicitly."
            ) from exc
        if not split_names:
            raise RuntimeError("No dataset splits found.")
        selected_split = next(
            (candidate for candidate in DEFAULT_SPLIT_CANDIDATES if candidate in split_names),
            split_names[0],
        )

    return load_dataset(dataset_id, name=dataset_config, split=selected_split, cache_dir=cache_dir), selected_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a seeded BigCodeBench-Hard problem set.")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("SEED", "42")),
        help="Random seed (default from SEED env).",
    )
    parser.add_argument("--count", type=int, default=None, help="Number of problems to generate.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Alias for --count (number of problems to generate).",
    )
    parser.add_argument("--out-dir", default="problems", help="Output directory for JSON files.")
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), remove existing problem_*.json files in out-dir before writing.",
    )
    parser.add_argument(
        "--dataset-id",
        default="bigcode/bigcodebench-hard",
        help="Hugging Face dataset ID for BigCodeBench-Hard.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional dataset config name, if required by the dataset.",
    )
    parser.add_argument(
        "--split",
        default="auto",
        help="Dataset split name (default: auto-detect from available splits).",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face datasets cache directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count is None and args.k is None:
        count = 10
    elif args.count is not None and args.k is not None and args.count != args.k:
        raise ValueError("--count and --k were both provided with different values.")
    else:
        count = args.count if args.count is not None else args.k

    if count is None or count <= 0:
        raise ValueError("Number of problems must be positive. Use --k or --count.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for old in out_dir.glob("problem_*.json"):
            old.unlink()
        manifest_old = out_dir / "manifest.json"
        if manifest_old.exists():
            manifest_old.unlink()

    dataset, selected_split = load_hf_dataset(
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    dataset_len = len(dataset)
    if count > dataset_len:
        raise ValueError(
            f"Requested problems ({count}) is larger than dataset split size ({dataset_len})."
        )

    rng = random.Random(args.seed)
    selected_indexes = rng.sample(range(dataset_len), k=count)
    manifest = []

    id_keys = ("task_id", "problem_id", "id", "question_id")
    title_keys = ("title", "task_name", "entry_point", "name")
    difficulty_keys = ("difficulty",)
    prompt_keys = ("complete_prompt", "prompt", "instruct_prompt", "question", "problem", "description")
    solution_keys = (
        "canonical_solution",
        "reference_solution",
        "solution",
        "official_solution",
        "answer",
        "answers",
        "solutions",
    )

    for i, dataset_index in enumerate(selected_indexes, start=1):
        row = dict(dataset[dataset_index])
        source_problem_id = first_non_empty(row, id_keys)
        title = first_non_empty(row, title_keys) or f"BigCodeBench-Hard Problem {dataset_index}"
        difficulty = first_non_empty(row, difficulty_keys) or "hard"
        prompt = first_non_empty(row, prompt_keys)
        canonical_solution = first_non_empty(row, solution_keys)

        if prompt is None:
            raise ValueError(
                "No prompt-like field found for dataset row "
                f"{dataset_index}. Expected one of: {', '.join(prompt_keys)}"
            )

        problem = {
            "problem_id": i,
            "seed": args.seed,
            "title": as_text(title),
            "difficulty": as_text(difficulty),
            "prompt": as_text(prompt),
            "canonical_solution": as_text(canonical_solution),
            "metadata": {
                "source": "bigcodebench-hard",
                "dataset_id": args.dataset_id,
                "dataset_config": args.dataset_config,
                "split": selected_split,
                "dataset_index": dataset_index,
                "source_problem_id": as_text(source_problem_id),
            },
        }

        path = out_dir / f"problem_{i:03d}.json"
        path.write_text(json.dumps(problem, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest.append(
            {
                "problem_id": i,
                "path": str(path),
                "difficulty": problem["difficulty"],
                "dataset_index": dataset_index,
                "source_problem_id": problem["metadata"]["source_problem_id"],
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"generated {count} problems from {args.dataset_id} "
        f"(split={selected_split}) in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
