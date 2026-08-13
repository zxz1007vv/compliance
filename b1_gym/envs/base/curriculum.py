"""Compatibility imports for the historical curriculum module path."""

from b1_gym.curriculum.curriculum import (
    Curriculum,
    RewardThresholdCurriculum,
    SumCurriculum,
    is_met,
    key_is_met,
)

__all__ = [
    "Curriculum",
    "RewardThresholdCurriculum",
    "SumCurriculum",
    "is_met",
    "key_is_met",
]
