#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from datasets import load_dataset
from tqdm import tqdm

from vera.aime import (
    AIMEAugmentor,
    GenerationConfig,
    ProgressLogger,
    SeedProblem,
)
from vera.aime.models import GeneratorArtifact, VariantOutcome
from vera.dataset_io import load_jsonl, save_jsonl
from vera.oracles import load_impl
from vera.oracle_llm_io import judge_llm_call
from vera.prompt_templates import hardest_variant_messages
from vera.utils import set_seed


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def _load_seed_problems(
    dataset_name: str,
    split: str,
    min_year: Optional[int],
    max_year: Optional[int],
    limit: Optional[int],
    dataset_format: str,
) -> List[SeedProblem]:
    ds = load_dataset(dataset_name, split=split)
    seeds: List[SeedProblem] = []

    for idx, row in enumerate(ds):
        if dataset_format == "aime":
            year = int(row["Year"])
            if min_year is not None and year < min_year:
                continue
            if max_year is not None and year > max_year:
                continue
            raw_id = str(row["ID"])
            question = str(row["Question"]).strip()
            answer = str(row["Answer"]).strip()
        elif dataset_format == "amo-bench":
            raw_id = str(row.get("question_id") or row.get("id") or idx)
            question = str(row.get("prompt") or row.get("question") or "").strip()
            answer = str(row.get("solution") or row.get("answer") or "").strip()
            year = int(row.get("year") or row.get("Year") or (max_year or 0))
        elif dataset_format == "beyond-aime":
            raw_id = str(row.get("ID") or row.get("problem_id") or idx)
            question = str(row.get("problem") or row.get("prompt") or "").strip()
            answer = str(row.get("answer") or row.get("asnwer") or "").strip()
            year = int(row.get("year") or row.get("Year") or (max_year or 0))
        else:
            raise ValueError(f"Unsupported dataset_format: {dataset_format}")

        if not question or not answer:
            continue

        if dataset_format == "aime":
            seed_id = f"{raw_id}::{year}"
        else:
            seed_id = str(raw_id)

        seeds.append(
            SeedProblem(
                id=seed_id,
                year=year,
                question=question,
                answer=answer,
            )
        )

    seeds.sort(key=lambda s: (s.year, s.id))
    if limit is not None:
        seeds = seeds[:limit]
    return seeds


def _variant_to_record(
    seed: SeedProblem,
    variant: VariantOutcome,
    source_id: str,
    config: GenerationConfig,
) -> Dict[str, object]:
    variant_id = f"{variant.generator_id}_s{variant.sample_index:02d}"
    judge_block = {
        "consistent": variant.judge_consistent,
        "successes": variant.judge_successes,
        "threshold": config.judge_consistency_threshold,
        "trials": [
            {
                "trial_index": t.trial_index,
                "provided_answer": t.provided_answer,
                "expected_truth": t.expected_truth,
                "parsed_decision": t.parsed_decision,
                "raw_decision": t.raw_decision,
                "is_noise": t.is_noise,
            }
            for t in variant.judge_trials
        ],
    }
    return {
        "seed_id": seed.id,
        "seed_original_id": source_id,
        "seed_year": seed.year,
        "variant_id": variant_id,
        "prompt_attempt": variant.prompt_attempt,
        "generator_id": variant.generator_id,
        "question": variant.question_text,
        "answer": variant.correct_answer,
        "numeric_answer": variant.numeric_answer,
        "assignment": variant.assignment,
        "judge": judge_block,
        "generator": {
            "attempts": variant.generator_attempts,
            "elapsed_seconds": variant.generator_elapsed_sec,
            "hardness_rationale": variant.metadata.get("hardness_rationale"),
            "teacher_notes": variant.metadata.get("teacher_notes"),
        },
        "noise_answers": variant.noise_answers,
        "metadata": variant.metadata,
        "source_question": seed.question,
        "source_answer": seed.answer,
    }


def _artifact_to_dict(artifact: GeneratorArtifact, seed: SeedProblem, source_id: str) -> Dict[str, object]:
    return {
        "generator_id": artifact.generator_id,
        "seed_id": artifact.seed_id,
        "seed_original_id": source_id,
        "seed_year": seed.year,
        "language_wrapper": artifact.language_wrapper,
        "combined_code": artifact.combined_code,
        "teacher_generator_code": artifact.teacher_generator_code,
        "teacher_verifier_code": artifact.teacher_verifier_code,
        "hardness_rationale": artifact.hardness_rationale,
        "notes": artifact.notes,
        "metadata": artifact.metadata,
        "source_question": seed.question,
        "source_answer": seed.answer,
    }


def _resolve_judge(judge_impl: str) -> Callable[[List[Dict[str, str]]], str]:
    obj = load_impl(judge_impl)
    if callable(obj):
        return obj
    if hasattr(obj, "decide") and callable(obj.decide):
        return obj.decide  # type: ignore[return-value]
    raise ValueError(f"Judge implementation {judge_impl} is not callable.")


def _extract_json_object(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except Exception:
        pass
    import re

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Failed to locate JSON object in judge response.")
    return json.loads(match.group(0))


def _flush_partial_augmented(out_path: str, buffer: List[Dict[str, object]]) -> None:
    """
    Persist currently collected augmented variants so progress isn't lost on failure.
    """
    if not buffer:
        return
    save_jsonl(out_path, buffer)
    print(f"[checkpoint] Wrote {len(buffer)} augmented variants to {out_path}", flush=True)


def _select_hardest_variant(
    seed_id: str,
    variants: List[Dict[str, object]],
    judge_call: Callable[[List[Dict[str, str]]], str],
) -> str:
    payload = [
        {
            "variant_id": item["variant_id"],
            "question": item["question"],
            "hardness_rationale": (item.get("generator") or {}).get("hardness_rationale")
            or (item.get("metadata") or {}).get("hardness_rationale")
            or "",
            "prompt_attempt": item.get("prompt_attempt"),
            "generator_attempts": (item.get("generator") or {}).get("attempts"),
        }
        for item in variants
    ]
    messages = hardest_variant_messages(seed_id, payload)
    raw = judge_call(messages)
    try:
        obj = _extract_json_object(raw)
    except Exception:
        return variants[0]["variant_id"]
    return str(obj.get("hardest_variant") or variants[0]["variant_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Augment AIME problems with harder parametric variants.")
    ap.add_argument("--teacher_impl", type=str, default="gsmk_aug.oracles:PromptTeacher")
    ap.add_argument("--judge_impl", type=str, default="gsmk_aug.oracle_llm_io:judge_llm_call")
    ap.add_argument("--dataset_name", type=str, default="di-zhang-fdu/AIME_1983_2024")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--dataset_format", type=str, choices=("aime", "amo-bench", "beyond-aime"), default="aime",
                    help="Controls how to interpret dataset fields.")
    ap.add_argument("--min_year", type=int, default=None)
    ap.add_argument("--max_year", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants_per_seed", type=int, default=5)
    ap.add_argument("--prompt_attempt_limit", type=int, default=20)
    ap.add_argument("--samples_per_prompt", type=int, default=5)
    ap.add_argument("--judge_consistency_threshold", type=int, default=4)
    ap.add_argument("--judge_correct_trials", type=int, default=2)
    ap.add_argument("--judge_noise_trials", type=int, default=3)
    ap.add_argument("--generator_timeout", type=float, default=10.0)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--out_augmented", type=str, required=True)
    ap.add_argument("--out_augmented_hard", type=str, required=True)
    ap.add_argument("--generators_dir", type=str, required=True)
    ap.add_argument("--progress_dir", type=str, default=None)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument(
        "--resume_from_augmented",
        type=str,
        default=None,
        help="Optional JSONL file containing prior augmented variants. Seeds already present there are skipped, and the old variants are merged into the outputs.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for deterministic ordering.")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = ap.parse_args()

    set_seed(args.seed)

    seeds = _load_seed_problems(
        dataset_name=args.dataset_name,
        split=args.split,
        min_year=args.min_year,
        max_year=args.max_year,
        limit=args.limit,
        dataset_format=args.dataset_format,
    )
    if not seeds:
        raise SystemExit("No seed problems found for the specified year range.")

    teacher = load_impl(args.teacher_impl)
    judge_call = _resolve_judge(args.judge_impl) if args.judge_impl else judge_llm_call

    config = GenerationConfig(
        variants_per_seed=args.variants_per_seed,
        prompt_attempt_limit=args.prompt_attempt_limit,
        samples_per_prompt=args.samples_per_prompt,
        generator_timeout_sec=args.generator_timeout,
        judge_consistency_threshold=args.judge_consistency_threshold,
        judge_correct_trials=args.judge_correct_trials,
        judge_noise_trials=args.judge_noise_trials,
        base_seed=args.base_seed,
        debug=args.debug,
    )

    augmentor = AIMEAugmentor(teacher=teacher, judge_call=judge_call, config=config)

    os.makedirs(os.path.dirname(args.out_augmented), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_augmented_hard), exist_ok=True)
    os.makedirs(args.generators_dir, exist_ok=True)
    if args.progress_dir:
        os.makedirs(args.progress_dir, exist_ok=True)
    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    all_variants: List[Dict[str, object]] = []
    variants_by_seed: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    summaries: List[Dict[str, object]] = []
    skipped_seed_ids: set[str] = set()

    if args.resume_from_augmented:
        if not os.path.exists(args.resume_from_augmented):
            print(f"[resume] Provided resume file {args.resume_from_augmented} does not exist; starting fresh.", flush=True)
        else:
            existing_rows = load_jsonl(args.resume_from_augmented)
            if existing_rows:
                all_variants.extend(existing_rows)
                for row in existing_rows:
                    variants_by_seed[row["seed_id"]].append(row)
                skipped_seed_ids = set(variants_by_seed.keys())
                print(
                    f"[resume] Loaded {len(existing_rows)} variants across {len(skipped_seed_ids)} seeds from {args.resume_from_augmented}.",
                    flush=True,
                )
            else:
                print(f"[resume] Resume file {args.resume_from_augmented} was empty.", flush=True)

    progress_bar = tqdm(seeds, desc="Augmenting AIME seeds", unit="problem")
    for idx, seed in enumerate(progress_bar, start=1):
        if seed.id in skipped_seed_ids:
            print(f"[{idx}/{len(seeds)}] Skipping {seed.id} (already present in resume file).", flush=True)
            continue
        source_id = seed.id.split("::", 1)[0]
        logger = ProgressLogger()
        progress_bar.set_postfix_str(seed.id)
        print(f"[{idx}/{len(seeds)}] Augmenting {seed.id}...", flush=True)
        result = augmentor.augment_seed(seed, logger=logger)

        for artifact in result.generators:
            out_path = os.path.join(
                args.generators_dir,
                _safe_filename(artifact.generator_id) + ".json",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(_artifact_to_dict(artifact, seed, source_id), f, ensure_ascii=False, indent=2)

        for variant in result.variants:
            record = _variant_to_record(seed, variant, source_id, config)
            all_variants.append(record)
            variants_by_seed[seed.id].append(record)

        summaries.append(
            {
                "seed_id": seed.id,
                "seed_original_id": source_id,
                "num_variants": len(result.variants),
                "valid_variants": sum(1 for v in result.variants if v.judge_consistent),
                "fallback_used": any(getattr(v, "metadata", {}).get("fallback") for v in result.variants),
                "prompt_attempts": result.summary.total_prompt_attempts,
                "samples_attempted": result.summary.total_samples,
                "failures": result.summary.failures,
            }
        )

        if args.progress_dir:
            progress_path = os.path.join(args.progress_dir, _safe_filename(seed.id) + "_progress.json")
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump([entry.to_dict() for entry in logger.entries], f, ensure_ascii=False, indent=2)

        print(f"[{idx}/{len(seeds)}] Completed augmentation for {seed.id} with {len(result.variants)} variants.")

        if idx % 5 == 0:
            _flush_partial_augmented(args.out_augmented, all_variants)

    hard_variants: List[Dict[str, object]] = []
    for seed_id, items in variants_by_seed.items():
        if not items:
            continue
        hardest_id = _select_hardest_variant(seed_id, items, judge_call)
        chosen = next((item for item in items if item["variant_id"] == hardest_id), items[0])
        hard_variants.append(chosen)

    save_jsonl(args.out_augmented, all_variants)
    save_jsonl(args.out_augmented_hard, hard_variants)

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_variants)} augmented variants to {args.out_augmented}")
    print(f"Wrote {len(hard_variants)} hardest variants to {args.out_augmented_hard}")


if __name__ == "__main__":
    main()
