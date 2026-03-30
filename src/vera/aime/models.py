from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SeedProblem:
    """Normalized representation of a seed AIME problem."""

    id: str
    year: int
    question: str
    answer: str


@dataclass
class TeacherSpec:
    """
    Structured response from the teacher LLM.

    The teacher still returns separate generator / validator snippets so the
    pipeline can orchestrate retries, logging, and judge validation, while
    downstream artifacts expose a combined generator interface.
    """

    seed_id: str
    language_wrapper: str
    generator_code: str
    verifier_code: str
    hardness_rationale: str
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """
    Tunable knobs for the augmentation workflow.
    """

    variants_per_seed: int = 5
    prompt_attempt_limit: int = 20
    samples_per_prompt: int = 5
    generator_timeout_sec: float = 10.0
    judge_consistency_threshold: int = 4
    judge_correct_trials: int = 2
    judge_noise_trials: int = 3
    # Deterministic shuffling is handled upstream; keep RNG seed for hashing.
    base_seed: int = 0
    debug: bool = False


@dataclass
class JudgeTrial:
    """One judge-LLM evaluation."""

    trial_index: int
    provided_answer: str
    expected_truth: bool
    raw_decision: str
    parsed_decision: Optional[bool]
    is_noise: bool


@dataclass
class VariantOutcome:
    """Captures the result of a single generator sample (after judge filtering)."""

    seed_id: str
    generator_id: str
    prompt_attempt: int
    sample_index: int
    assignment: Dict[str, Any]
    question_text: str
    correct_answer: str
    numeric_answer: Optional[float]
    generator_attempts: int
    generator_elapsed_sec: float
    judge_trials: List[JudgeTrial] = field(default_factory=list)
    judge_consistent: bool = False
    judge_successes: int = 0
    noise_answers: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorArtifact:
    """Final combined generator export."""

    generator_id: str
    seed_id: str
    language_wrapper: str
    combined_code: str
    teacher_generator_code: str
    teacher_verifier_code: str
    hardness_rationale: str
    notes: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationSummary:
    """Aggregated stats for reporting and logging."""

    seed_id: str
    total_prompt_attempts: int
    total_samples: int
    valid_variants: int
    failures: List[str] = field(default_factory=list)
