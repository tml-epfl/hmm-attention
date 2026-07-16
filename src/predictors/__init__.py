from typing import Optional

from src.predictors.base import Predictor
from src.predictors.classification import (
    ClassificationPredictor,
    HierarchicalPredictor,
)
from src.predictors.regression import RegressionPredictor

__all__ = [
    "Predictor",
    "ClassificationPredictor",
    "HierarchicalPredictor",
    "RegressionPredictor",
    "build_predictor",
]


def build_predictor(
    teacher,
    kind: str = "classification",
    argmax: bool = False,
    noise_std: float = 0.0,
) -> Predictor:
    """Wrap a teacher in a Predictor.

    `kind` selects between classification (discrete one-hot) and regression
    (continuous). For classification, `HierarchicalTeacher` gets a
    `HierarchicalPredictor` so burn-in emits valid chunk-composed prefixes;
    every other teacher uses the plain `ClassificationPredictor`.
    """
    # Local import keeps the package importable without materializing the whole
    # teacher hierarchy at module load.
    from src.teachers import HierarchicalTeacher

    if kind == "classification":
        if isinstance(teacher, HierarchicalTeacher):
            return HierarchicalPredictor(teacher, argmax=argmax)
        return ClassificationPredictor(teacher, argmax=argmax)
    if kind == "regression":
        return RegressionPredictor(teacher, noise_std=noise_std)
    raise ValueError(f"Unknown predictor kind: {kind!r}")
