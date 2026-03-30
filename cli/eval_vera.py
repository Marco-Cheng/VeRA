#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from datasets import load_dataset

from vera.aime.evaluation import (
    ProblemResult,
    aggregate_results,
    evaluate_problem,
)
from vera.dataset_io import load_jsonl
from vera.oracles import load_impl
from vera.utils import set_seed


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _load_seed_items(
    dataset_name: str,
    split: str,
    min_year: int,
    max_year: int,
    limit: Optional[int],
) -> List[Dict[str, str]]:
    ds = load_dataset(dataset_name, split=split)
    items: List[Dict[str, str]] = []
    for row in ds:
        year = int(row["Year"])
        if year < min_year or year > max_year:
            continue
        item_id = str(row["ID"])
        question = str(row["Question"]).strip()
        answer = str(row["Answer"]).strip()
        items.append(
            {
                "id": f"{item_id}::{year}",
                "question": question,
                "answer": answer,
                "answer_value": _parse_float(answer),
            }
        )
    items.sort(key=lambda x: x["id"])
    if limit is not None:
        items = items[:limit]
    return items


def _load_augmented_items(path: str) -> List[Dict[str, str]]:
    rows = load_jsonl(path)
    items: List[Dict[str, str]] = []
    for i, row in enumerate(rows):
        variant_id = row.get("variant_id") or row.get("id") or f"variant_{i}"
        question = row.get("question") or ""
        answer = str(row.get("answer", "")).strip()
        numeric = row.get("numeric_answer")
        items.append(
            {
                "id": str(variant_id),
                "question": question,
                "answer": answer,
                "answer_value": float(numeric) if isinstance(numeric, (int, float)) else _parse_float(answer),
            }
        )
    return items


def _problem_result_to_dict(res: ProblemResult) -> Dict[str, object]:
    return {
        "item_id": res.item_id,
        "question": res.question,
        "gold_answer": res.gold_answer,
        "gold_value": res.gold_value,
        "avg_at_k": res.avg_at_k,
        "pass_at_k": res.pass_at_k,
        "majority_vote_correct": res.majority_vote_correct,
        "majority_vote_answer": res.majority_vote_answer,
        "majority_vote_value": res.majority_vote_value,
        "runs": [
            {
                "run_index": run.run_index,
                "raw_response": run.raw_response,
                "final_chunk": run.final_chunk,
                "final_value": run.final_value,
                "correct": run.correct,
            }
            for run in res.runs
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate student models on AIME datasets.")
    ap.add_argument("--student_impl", type=str, required=True)
    ap.add_argument("--dataset_mode", choices=("seed", "augmented", "augmented-hard"), required=True)
    ap.add_argument("--dataset_path", type=str, help="Path to augmented JSONL when using augmented modes.")
    ap.add_argument("--dataset_name", type=str, default="di-zhang-fdu/AIME_1983_2024")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--min_year", type=int, help="Inclusive minimum year for seed mode.")
    ap.add_argument("--max_year", type=int, help="Inclusive maximum year for seed mode.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of items.")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=1e-3)
    ap.add_argument("--report_json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.dataset_mode == "seed":
        if args.min_year is None or args.max_year is None:
            raise SystemExit("--min_year and --max_year are required for seed mode.")
        items = _load_seed_items(
            dataset_name=args.dataset_name,
            split=args.split,
            min_year=args.min_year,
            max_year=args.max_year,
            limit=args.limit,
        )
    else:
        if not args.dataset_path:
            raise SystemExit("--dataset_path is required for augmented modes.")
        items = _load_augmented_items(args.dataset_path)
        if args.limit is not None:
            items = items[: args.limit]

    student = load_impl(args.student_impl)

    total_items = len(items)
    print(f"Evaluating {total_items} item(s)...", flush=True)

    results: List[ProblemResult] = []
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{total_items}] Evaluating {item['id']}...", flush=True)
        res = evaluate_problem(
            item_id=item["id"],
            question=item["question"],
            gold_answer=item["answer"],
            gold_value=item["answer_value"],
            student=student,
            runs=args.runs,
            tol=args.tolerance,
        )
        results.append(res)

    aggregate = aggregate_results(results)
    print(f"Evaluated {len(results)} items.")
    print(f"avg@{args.runs}: {aggregate['avg_at_k']:.4f}")
    print(f"pass@{args.runs}: {aggregate['pass_at_k']:.4f}")
    print(f"majority@{args.runs}: {aggregate['majority_vote']:.4f}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_mode": args.dataset_mode,
                    "dataset_path": args.dataset_path,
                    "runs": args.runs,
                    "tolerance": args.tolerance,
                    "num_items": len(results),
                    "metrics": aggregate,
                    "problems": [_problem_result_to_dict(res) for res in results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


if __name__ == "__main__":
    main()
