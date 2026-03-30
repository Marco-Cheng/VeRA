"""
High-level exports for the AIME augmentation toolkit.
"""

from .aime import (  # noqa: F401
    AIMEAugmentor,
    AugmentationResult,
    AugmentationSummary,
    GeneratorArtifact,
    GenerationConfig,
    JudgeTrial,
    ProgressLogger,
    SeedProblem,
    TeacherSpec,
    VariantOutcome,
)
from .dataset_io import load_jsonl, save_jsonl  # noqa: F401
from .oracle_llm_io import judge_llm_call, student_llm_call, teacher_llm_call  # noqa: F401
from .oracles import (  # noqa: F401
    PromptStudent,
    PromptTeacher,
    StudentOracle,
    TeacherOracle,
    load_impl,
)
from .prompt_templates import (  # noqa: F401
    aime_teacher_messages,
    hardest_variant_messages,
    judge_messages,
    student_messages,
    teacher_messages,
)
from .utils import set_seed  # noqa: F401
from .verifier import RNGShim, compile_generator, compile_verifier, run_verifier  # noqa: F401
