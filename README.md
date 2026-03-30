# VeRA —  Verified Reasoning Augmentation at Scale

VeRA turns seed math problems into richer evaluation suites by stacking three difficulty levels:

- **vera-E (Equivalence)** – faithful rephrasings of the seed logic (e.g. alternate wording, translation).
- **vera-H (Hardness)** – parameterized variants that preserve the idea but increase reasoning depth.
- **vera-H Pro** – for every seed, the single hardest vera-H variant chosen by a judge LLM.

This repository contains the end-to-end pipeline, progress logs, and all datasets already collected for AIME, BeyondAIME, and Meituan’s AMO-Bench.

---

## Layout

```
VeRA-final/
├── README.md
├── pyproject.toml / requirements.txt
├── cli/
│   ├── prepare_vera.py      # augment seeds → vera-E/H/H Pro artifacts
│   └── eval_vera.py         # evaluate student models on any split
├── src/vera/                # reusable library (augmentors, prompts, judge logic, utils)
├── artifacts/
    ├── vera_dataset/        # generated datasets for evaluation, main artefact of the repo
        ├── seeds/
│       ├── VeRA-E/              
│       ├── VeRA-H/
│       ├── VeRA-H-Pro/
│       ├── VeRA-H verified/
│       ├── VeRA-H-Pro verified/
│   ├── analysis/            # pass-rate studies, low-pass audits
│   ├── reports/             # augmentation summaries
│   └── logs/                # generator binaries + per-seed progress traces
└── ...
```

All Python modules live under `src/vera`, so after installing (`pip install -e .`) you can import `vera.*` anywhere.

---

## Running the pipeline

### 1. Install

```bash
cd VeRA-final
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 2. Configure LLM backends

`src/vera/oracle_llm_io.py` expects the same ByteDance endpoints used previously. Set these env vars before running:

```bash
export AIME_TEACHER_URL="https://..."
export AIME_STUDENT_URL="https://..."
export AIME_JUDGE_URL="https://..."
export AIME_LOG_LLM_IO=1   # optional verbose IO audit
```

### 3. Augment seeds (vera-E + vera-H + vera-H Pro)

`cli/prepare_vera.py` unifies all datasets. Pick the format that matches your HuggingFace split.

#### AIME (default format)
```bash
python3 cli/prepare_vera.py \
  --teacher_impl vera.oracles:PromptTeacher \
  --judge_impl vera.oracle_llm_io:judge_llm_call \
  --dataset_name di-zhang-fdu/AIME_1983_2024 \
  --dataset_format aime \
  --variants_per_seed 5 \
  --out_augmented artifacts/vera-H/aime24_augmented.jsonl \
  --out_augmented_hard artifacts/vera-H-Pro/aime24_augmented_hard.jsonl \
  --generators_dir artifacts/logs/generators \
  --progress_dir artifacts/logs/progress \
  --summary_json artifacts/reports/aime24_summary.json
```

#### BeyondAIME
```bash
python3 cli/prepare_vera.py \
  --teacher_impl vera.oracles:PromptTeacher \
  --judge_impl vera.oracle_llm_io:judge_llm_call \
  --dataset_name ByteDance-Seed/BeyondAIME \
  --dataset_format beyond-aime \
  --variants_per_seed 5 \
  --out_augmented artifacts/vera-H/beyond_aime_augmented.jsonl \
  --out_augmented_hard artifacts/vera-H-Pro/beyond_aime_augmented_hard.jsonl \
  --summary_json artifacts/reports/beyond_aime_summary.json
```

#### AMO-Bench (Meituan Longcat)
```bash
python3 cli/prepare_vera.py \
  --teacher_impl vera.oracles:PromptTeacher \
  --judge_impl vera.oracle_llm_io:judge_llm_call \
  --dataset_name meituan-longcat/AMO-Bench \
  --dataset_format amo-bench \
  --variants_per_seed 5 \
  --out_augmented artifacts/vera-H/amo_bench_augmented.jsonl \
  --out_augmented_hard artifacts/vera-H-Pro/amo_bench_augmented_hard.jsonl \
  --summary_json artifacts/reports/amo_bench_summary.json
```

Key flags:

- `--dataset_format` selects how columns are read:
  - `aime`: `ID`, `Question`, `Answer`, `Year`.
  - `beyond-aime`: `problem`, `answer/asnwer`.
  - `amo-bench`: `question_id`, `prompt`, `solution`.
- `--variants_per_seed`: number of accepted vera-H variants per seed.
- `--debug`: enables ultra-verbose logging (teaches, generator attempts, judge context).
- Progress is printed via `tqdm` *and* checkpoints every 5 seeds so `out_augmented` is always re-writable.

### 4. Evaluate models

```bash
python3 cli/eval_vera.py \
  --student_impl vera.oracles:PromptStudent \
  --dataset_mode augmented \
  --dataset_path artifacts/vera-H/aime24_augmented.jsonl \
  --runs 5 \
  --tolerance 1e-3 \
  --report_json artifacts/analysis/aime24_eval_report.json
```

`dataset_mode` accepts `seed`, `augmented`, or `augmented-hard` (vera-H Pro). The evaluator enforces the “last token is the answer” rule with ±1e−3 tolerance and reports `avg@k`, `pass@k`, and `majority@k`.

---

## Delivered datasets

- **vera-E (artifacts/vera_datasets/vera-E)**

- **vera-H (artifacts/vera_datasets/vera-H)**

- **vera-H Pro (artifacts/vera_datasets/vera-H-Pro)**

Raw generator code for each prompt attempt resides in `artifacts/logs/generators/<seed>.json`. Progress logs with judge verdicts are in `artifacts/logs/progress/`.

---

## Next steps for customization

1. Plug in your teacher/student/judge models via `src/vera/oracle_llm_io.py`.
2. Regenerate vera-E/H/H Pro for any new seed ranges or datasets.
3. Use `cli/eval_vera.py` to benchmark student LLMs on each level and compare generalization gaps.
