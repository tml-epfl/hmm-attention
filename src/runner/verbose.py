import logging

import torch

from src.teachers import LinearARTeacher


def log_teacher_summary(teacher: torch.nn.Module) -> None:
    """Log teacher weight stats. No-op for teachers that aren't `LinearARTeacher`."""
    if not isinstance(teacher, LinearARTeacher):
        return
    _log_summary("Teacher", teacher, teacher._params)


def log_student_summary(student: torch.nn.Module) -> None:
    """Log student weight stats if the student happens to be a `LinearARTeacher`."""
    if not isinstance(student, LinearARTeacher):
        return
    _log_summary("Student", student, student._get_weights())


def _log_summary(role: str, model: LinearARTeacher, params: torch.Tensor) -> None:
    logger = logging.getLogger()
    logger.info(f"===== {role} =====")
    logger.info(f"{role} rank: {model.rank}")
    logger.info(f"{role} dim: {model.dim}")
    logger.info(f"{role} window: {model.window}")
    logger.info(f"{role} scale: {model.scale}")
    logger.info(f"{role} weights: {model._get_weights().shape}")

    flat = params.view(-1, params.size(-1))
    logger.info(
        f"Frobenius norm/norm^2: {torch.linalg.norm(flat)}, "
        f"{torch.linalg.norm(flat) ** 2}"
    )
    logger.info(
        f"Operator norm/norm^2: {torch.linalg.norm(flat, ord=2)}, "
        f"{torch.linalg.norm(flat, ord=2) ** 2}"
    )
