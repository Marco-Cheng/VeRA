from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .models import (
    AugmentationSummary,
    GenerationConfig,
    GeneratorArtifact,
    JudgeTrial,
    SeedProblem,
    TeacherSpec,
    VariantOutcome,
)
from .progress import ProgressLogger, ProgressEntry
from ..prompt_templates import aime_teacher_messages, judge_messages
from ..verifier import RNGShim, compile_generator, compile_verifier, run_verifier


JudgeCaller = Callable[[List[Dict[str, str]]], str]


def _hash_seed(*parts: str, base_seed: int = 0) -> int:
    key = "|".join(parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") ^ base_seed) & 0x7FFFFFFF


def _render_wrapper(wrapper: str, assignment: Dict[str, object]) -> str:
    def replace_brace(text: str) -> str:
        out = text
        for name in set(_brace_names(text)):
            out = out.replace("{" + name + "}", str(assignment.get(name, f"{{{name}}}")))
        for name in set(_bracket_names(text)):
            out = out.replace("[" + name + "]", str(assignment.get(name, f"[{name}]")))
        return out

    return replace_brace(wrapper)


def _brace_names(text: str) -> Iterable[str]:
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            return
        j = text.find("}", i + 1)
        if j == -1:
            return
        yield text[i + 1 : j]
        start = j + 1


def _bracket_names(text: str) -> Iterable[str]:
    start = 0
    while True:
        i = text.find("[", start)
        if i == -1:
            return
        j = text.find("]", i + 1)
        if j == -1:
            return
        yield text[i + 1 : j]
        start = j + 1


def _ensure_str_answer(answer: object) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (int, float)):
        if isinstance(answer, int):
            return str(answer)
        return f"{answer:.12g}"
    return json.dumps(answer, ensure_ascii=False)


def _parse_bool(text: str) -> Optional[bool]:
    if not text:
        return None
    matches = re.findall(r"(true|false)", text, flags=re.IGNORECASE)
    if not matches:
        return None
    last = matches[-1].lower()
    if last == "true":
        return True
    if last == "false":
        return False
    return None


def _generate_noise_answers(correct_ans: str, count: int, rng: random.Random) -> List[str]:
    try:
        val = float(correct_ans)
        is_int = abs(val - round(val)) < 1e-9
    except Exception:
        return [correct_ans + f" (incorrect variant {i+1})" for i in range(count)]

    noises: List[str] = []
    seen = {correct_ans}
    while len(noises) < count:
        if is_int:
            delta = rng.randint(1, 9)
            if rng.random() < 0.5:
                candidate = int(round(val)) + delta
            else:
                candidate = max(0, int(round(val)) - delta)
            cand_str = str(candidate)
        else:
            span = max(1.0, abs(val) * 0.1)
            delta = rng.uniform(0.05 * span, span)
            candidate = val + (delta if rng.random() < 0.5 else -delta)
            cand_str = f"{candidate:.6f}"
        if cand_str not in seen:
            seen.add(cand_str)
            noises.append(cand_str)
    return noises


def _build_combined_code(language_wrapper: str, generator_code: str, verifier_code: str) -> str:
    wrapper_repr = repr(language_wrapper)
    rng_shim = textwrap.dedent(
        """
        class _RNGShim:
            def __init__(self, seed: int):
                import random as _random
                self._seed = int(seed)
                self._r = _random.Random(self._seed)

            def random(self): return self._r.random()
            def uniform(self, a, b): return self._r.uniform(a, b)
            def randint(self, a, b): return self._r.randint(a, b)
            def randrange(self, *args): return self._r.randrange(*args)
            def choice(self, seq): return self._r.choice(seq)
            def choices(self, population, weights=None, cum_weights=None, k: int = 1):
                return self._r.choices(population, weights=weights, cum_weights=cum_weights, k=k)
            def sample(self, population, k: int): return self._r.sample(population, k)
            def shuffle(self, x):
                self._r.shuffle(x)
                return x
            def getrandbits(self, k: int): return self._r.getrandbits(k)
        """
    ).strip()
    combined = textwrap.dedent(
        f"""
        import math
        import random
        import time

        LANGUAGE_WRAPPER = {wrapper_repr}

        {rng_shim}

        {generator_code.strip()}

        {verifier_code.strip()}


        def generate_valid_instance(seed: int, timeout_seconds: float = 10.0):
            \"\"\"Return dict with assignment/question/answer or raise RuntimeError.\"\"\"
            start = time.time()
            attempts = 0
            while time.time() - start < timeout_seconds:
                attempts += 1
                rng = _RNGShim(seed + attempts)
                assignment = generator(rng)
                valid, answer = verifier(assignment)
                if valid:
                    question = LANGUAGE_WRAPPER.format(**assignment)
                    return {{
                        "assignment": assignment,
                        "question": question,
                        "answer": answer,
                        "attempts": attempts,
                        "elapsed_seconds": time.time() - start,
                    }}
            raise RuntimeError("Failed to sample a valid assignment within timeout_seconds")
        """
    ).strip()
    return combined + "\n"


def _parse_teacher_payload(seed: SeedProblem, payload: Dict[str, object]) -> TeacherSpec:
    language_wrapper = payload.get("language_wrapper")
    if not language_wrapper:
        text_templates = payload.get("text_templates") or []
        if text_templates:
            language_wrapper = text_templates[0]
    if not isinstance(language_wrapper, str) or not language_wrapper.strip():
        raise ValueError("Teacher response missing non-empty language_wrapper.")

    generator_block = payload.get("generator") or {}
    verifier_block = (
        payload.get("validator")
        or payload.get("verifier")
        or {}
    )
    gen_code = generator_block.get("code")
    ver_code = verifier_block.get("code")
    if not gen_code or not isinstance(gen_code, str):
        raise ValueError("Teacher response missing generator.code.")
    if not ver_code or not isinstance(ver_code, str):
        raise ValueError("Teacher response missing verifier/validator code.")

    rationale = payload.get("hardness_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "Teacher did not provide a hardness rationale."

    notes = payload.get("notes")
    metadata = payload.get("meta") or {}

    return TeacherSpec(
        seed_id=seed.id,
        language_wrapper=language_wrapper.strip(),
        generator_code=gen_code,
        verifier_code=ver_code,
        hardness_rationale=rationale.strip(),
        notes=notes.strip() if isinstance(notes, str) else None,
        metadata=dict(metadata),
    )


@dataclass
class AugmentationResult:
    seed: SeedProblem
    variants: List[VariantOutcome]
    generators: List[GeneratorArtifact]
    summary: AugmentationSummary
    progress: List[ProgressEntry]


class AIMEAugmentor:
    """
    Orchestrates teacher prompting, generator sampling, judge validation, and export artifacts.
    """

    def __init__(
        self,
        teacher,
        judge_call: JudgeCaller,
        config: GenerationConfig,
        logger: Optional[ProgressLogger] = None,
    ):
        self.teacher = teacher
        self.judge_call = judge_call
        self.config = config
        self.logger = logger

    def _debug(
        self,
        logger: ProgressLogger,
        stage: str,
        message: str,
        *,
        seed_id: Optional[str] = None,
        **payload: object,
    ) -> None:
        if self.config.debug:
            logger.log(seed_id or "DEBUG", f"debug::{stage}", message, **payload)

    def augment_seed(self, seed: SeedProblem, logger: Optional[ProgressLogger] = None) -> AugmentationResult:
        logger = logger or self.logger or ProgressLogger()
        self._debug(
            logger,
            "augment_seed",
            "Entered augment_seed.",
            seed_id=seed.id,
            seed_year=seed.year,
            question_preview=seed.question[:120],
            answer=seed.answer,
            config=dataclasses.asdict(self.config) if self.config.debug else None,
        )
        valid_variants: List[VariantOutcome] = []
        generator_artifacts: List[GeneratorArtifact] = []
        failures: List[str] = []
        feedback: Optional[str] = None

        for prompt_attempt in range(1, self.config.prompt_attempt_limit + 1):
            if len(valid_variants) >= self.config.variants_per_seed:
                break

            logger.log(
                seed.id,
                "teacher_prompt",
                f"Requesting augmentation spec (attempt {prompt_attempt}).",
                feedback=bool(feedback),
            )
            self._debug(
                logger,
                "teacher_prompt",
                "Calling teacher.convert_to_spec.",
                seed_id=seed.id,
                attempt=prompt_attempt,
                feedback=feedback,
                question=seed.question,
                answer=seed.answer,
                example_id=seed.id,
            )
            try:
                payload = self.teacher.convert_to_spec(
                    question=seed.question,
                    answer=seed.answer,
                    example_id=seed.id,
                    n_text_templates=1,
                    feedback=feedback,
                )
            except Exception as exc:
                msg = f"Teacher call failed: {exc}"
                failures.append(msg)
                feedback = f"Teacher call raised {type(exc).__name__}: {exc}"
                logger.log(seed.id, "teacher_error", msg)
                continue

            try:
                spec = _parse_teacher_payload(seed, payload)
            except Exception as exc:
                msg = f"Teacher spec parsing error: {exc}"
                failures.append(msg)
                feedback = (
                    "Teacher JSON was missing required fields. "
                    "Ensure language_wrapper, generator.code, and verifier.code are present."
                )
                logger.log(seed.id, "teacher_parse_error", msg)
                continue

            generator_id = f"{seed.id}_prompt{prompt_attempt:02d}"
            combined_code = _build_combined_code(
                spec.language_wrapper, spec.generator_code, spec.verifier_code
            )
            generator_artifacts.append(
                GeneratorArtifact(
                    generator_id=generator_id,
                    seed_id=seed.id,
                    language_wrapper=spec.language_wrapper,
                    combined_code=combined_code,
                    teacher_generator_code=spec.generator_code,
                    teacher_verifier_code=spec.verifier_code,
                    hardness_rationale=spec.hardness_rationale,
                    notes=spec.notes,
                    metadata=spec.metadata,
                )
            )

            try:
                gen_fn = compile_generator(spec.generator_code, time_limit_sec=int(self.config.generator_timeout_sec))
                ver_fn = compile_verifier(spec.verifier_code)
            except Exception as exc:
                msg = f"Compilation failed: {exc}"
                failures.append(msg)
                feedback = (
                    "Compilation failed for generator/verifier. Please ensure both are valid Python and "
                    "the generator returns dict assignments while verifier returns (bool, answer)."
                )
                logger.log(seed.id, "compile_error", msg)
                continue

            attempt_failures: List[str] = []
            for sample_idx in range(1, self.config.samples_per_prompt + 1):
                if len(valid_variants) >= self.config.variants_per_seed:
                    break

                seed_value = _hash_seed(seed.id, generator_id, str(sample_idx), base_seed=self.config.base_seed)
                variant = self._sample_variant(
                    seed,
                    spec,
                    generator_id,
                    prompt_attempt,
                    sample_idx,
                    gen_fn,
                    ver_fn,
                    seed_value,
                    logger,
                )
                if variant is None:
                    attempt_failures.append(f"Sample {sample_idx} failed to produce valid assignment.")
                    continue

                if variant.judge_consistent:
                    valid_variants.append(variant)
                    logger.log(
                        seed.id,
                        "variant_success",
                        f"Accepted variant {variant.generator_id}-{variant.sample_index}",
                        judge_successes=variant.judge_successes,
                    )
                else:
                    attempt_failures.append(
                        f"Judge rejected variant {variant.sample_index} "
                        f"(successes={variant.judge_successes})."
                    )

            if len(valid_variants) < self.config.variants_per_seed:
                if attempt_failures:
                    feedback = " | ".join(attempt_failures[:3])
                else:
                    feedback = (
                        "No valid variants were produced. Ensure generator finds harder assignments "
                        "consistent with verifier and judge feedback."
                    )

        if not valid_variants:
            logger.log(seed.id, "fallback", "Falling back to deterministic rephrasing.")
            valid_variants.extend(self._fallback_rephrase(seed))

        summary = AugmentationSummary(
            seed_id=seed.id,
            total_prompt_attempts=min(self.config.prompt_attempt_limit, len(generator_artifacts)),
            total_samples=len(generator_artifacts) * self.config.samples_per_prompt,
            valid_variants=len(valid_variants),
            failures=failures,
        )
        return AugmentationResult(
            seed=seed,
            variants=valid_variants,
            generators=generator_artifacts,
            summary=summary,
            progress=logger.entries,
        )

    def _sample_variant(
        self,
        seed: SeedProblem,
        spec: TeacherSpec,
        generator_id: str,
        prompt_attempt: int,
        sample_index: int,
        gen_fn,
        ver_fn,
        seed_value: int,
        logger: ProgressLogger,
    ) -> Optional[VariantOutcome]:
        start = time.time()
        attempts = 0

        self._debug(
            logger,
            "_sample_variant",
            "Entered _sample_variant.",
            seed_id=seed.id,
            generator_id=generator_id,
            prompt_attempt=prompt_attempt,
            sample_index=sample_index,
            seed_value=seed_value,
        )

        assignment = None
        answer = None
        while time.time() - start < self.config.generator_timeout_sec:
            attempts += 1
            try:
                rng = RNGShim(seed_value + attempts)
                candidate = gen_fn(rng)
            except Exception as exc:
                logger.log(
                    seed.id,
                    "generator_runtime_error",
                    f"Generator raised {exc}",
                    attempt=attempts,
                    sample_index=sample_index,
                )
                continue
            self._debug(
                logger,
                "generator_sample",
                "Generator produced candidate assignment.",
                seed_id=seed.id,
                attempt=attempts,
                assignment=candidate,
            )
            if not isinstance(candidate, dict):
                logger.log(
                    seed.id,
                    "generator_invalid_output",
                    "Generator did not return a dict assignment.",
                    attempt=attempts,
                    sample_index=sample_index,
                )
                continue

            ok, y = run_verifier(ver_fn, candidate)
            self._debug(
                logger,
                "verifier_result",
                "Verifier evaluated candidate.",
                seed_id=seed.id,
                attempt=attempts,
                valid=ok,
                answer=y,
            )
            if not ok:
                continue
            assignment = candidate
            answer = y
            break

        elapsed = time.time() - start
        if assignment is None:
            logger.log(
                seed.id,
                "generator_timeout",
                "No valid assignment within timeout.",
                sample_index=sample_index,
                elapsed=elapsed,
            )
            return None

        question_text = _render_wrapper(spec.language_wrapper, assignment)
        answer_str = _ensure_str_answer(answer)

        teacher_context = spec.hardness_rationale or ""
        if spec.notes:
            teacher_context = (teacher_context + "\nNotes: " + spec.notes).strip()
        judge_trials, successes, noise_answers = self._run_judge(
            question_text,
            answer_str,
            seed_value,
            logger=logger,
            seed_id=seed.id,
            teacher_context=teacher_context or None,
        )
        consistent = successes >= self.config.judge_consistency_threshold
        if not consistent:
            logger.log(
                seed.id,
                "judge_reject",
                "Judge deemed the candidate inconsistent with gold answer.",
                sample_index=sample_index,
                judge_successes=successes,
            )

        return VariantOutcome(
            seed_id=seed.id,
            generator_id=generator_id,
            prompt_attempt=prompt_attempt,
            sample_index=sample_index,
            assignment=assignment,
            question_text=question_text,
            correct_answer=answer_str,
            numeric_answer=self._to_float(answer_str),
            generator_attempts=attempts,
            generator_elapsed_sec=elapsed,
            judge_trials=judge_trials,
            judge_consistent=consistent,
            judge_successes=successes,
            noise_answers=noise_answers,
            metadata={
                "hardness_rationale": spec.hardness_rationale,
                "teacher_notes": spec.notes,
            },
        )

    def _run_judge(
        self,
        question_text: str,
        correct_answer: str,
        seed_value: int,
        logger: Optional[ProgressLogger] = None,
        seed_id: Optional[str] = None,
        teacher_context: Optional[str] = None,
    ) -> Tuple[List[JudgeTrial], int, List[str]]:
        trials: List[JudgeTrial] = []
        successes = 0
        rng = random.Random(seed_value)

        if logger is not None:
            self._debug(
                logger,
                "_run_judge",
                "Entered _run_judge.",
                seed_id=seed_id,
                question_preview=question_text[:120],
                correct_answer=correct_answer,
                seed_value=seed_value,
            )

        prompts: List[Tuple[str, bool, bool]] = []
        for _ in range(self.config.judge_correct_trials):
            prompts.append((correct_answer, True, False))
        noises = _generate_noise_answers(correct_answer, self.config.judge_noise_trials, rng)
        for noise in noises:
            prompts.append((noise, False, True))
        rng.shuffle(prompts)

        for idx, (candidate_answer, expected, is_noise) in enumerate(prompts, start=1):
            messages = judge_messages(question_text, candidate_answer, teacher_context=teacher_context)
            if logger is not None:
                self._debug(
                    logger,
                    "judge_trial",
                    "Submitting candidate answer to judge.",
                    seed_id=seed_id,
                    trial_index=idx,
                    candidate_answer=candidate_answer,
                    expected_truth=expected,
                    is_noise=is_noise,
                )
            raw = self.judge_call(messages)
            parsed = _parse_bool(raw or "")
            if parsed is not None and parsed == expected:
                successes += 1
            trials.append(
                JudgeTrial(
                    trial_index=idx,
                    provided_answer=candidate_answer,
                    expected_truth=expected,
                    raw_decision=raw or "",
                    parsed_decision=parsed,
                    is_noise=is_noise,
                )
            )

        return trials, successes, noises

    def _fallback_rephrase(self, seed: SeedProblem) -> List[VariantOutcome]:
        variants: List[VariantOutcome] = []
        templates = [
            f"In the {seed.year} AIME, contestants faced: {seed.question}",
            f"Rephrased challenge inspired by the {seed.year} AIME problem: {seed.question}",
            f"Consider this AIME-style task: {seed.question}",
            f"Alternate wording of the seed problem: {seed.question}",
            f"Restatement for clarity: {seed.question}",
        ]
        answer_str = _ensure_str_answer(seed.answer)
        for idx, text in enumerate(templates, start=1):
            variants.append(
                VariantOutcome(
                    seed_id=seed.id,
                    generator_id=f"{seed.id}_fallback",
                    prompt_attempt=0,
                    sample_index=idx,
                    assignment={},
                    question_text=text,
                    correct_answer=answer_str,
                    numeric_answer=self._to_float(answer_str),
                    generator_attempts=0,
                    generator_elapsed_sec=0.0,
                    judge_trials=[],
                    judge_consistent=True,
                    judge_successes=self.config.judge_consistency_threshold,
                    noise_answers=[],
                    metadata={"fallback": True},
                )
            )
        return variants[: self.config.variants_per_seed]

    @staticmethod
    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None
