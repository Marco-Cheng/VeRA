"""
Utilities for augmenting the AIME dataset with parametrized problem generators.

This subpackage contains dataclasses, prompt helpers, augmentation workflows,
and evaluation helpers tailored to the new AIME augmentation pipeline.
"""

from .models import (
    SeedProblem,
    TeacherSpec,
    GenerationConfig,
    GeneratorArtifact,
    JudgeTrial,
    VariantOutcome,
    AugmentationSummary,
)
from .progress import ProgressLogger
from .augmentation import AIMEAugmentor, AugmentationResult

__all__ = [
    "SeedProblem",
    "TeacherSpec",
    "GenerationConfig",
    "GeneratorArtifact",
    "JudgeTrial",
    "VariantOutcome",
    "AugmentationSummary",
    "ProgressLogger",
    "AIMEAugmentor",
    "AugmentationResult",
]
