"""Flow-GRPO adapters for Robust-U1 restoration rewards.

This module keeps the Robust-U1 reward implementation independent from
Flow-GRPO while exposing registry helpers that match Flow-GRPO reward names.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

from .rewards import structure_score, tinyclip_score

RewardFunction = Callable[..., tuple[list[float], dict[str, Any]]]
RewardBuilder = Callable[[Any], RewardFunction]

FLOW_GRPO_REFERENCE_REWARD_NAMES = frozenset(
    {
        "image_similarity",
        "restoration",
        "structure",
        "robust_structure",
        "tinyclip",
        "robust_tinyclip",
        "siglip",
    }
)


def restoration_score(device: Any) -> RewardFunction:
    """Flow-GRPO registry name for the Robust-U1 SSIM structure reward."""
    return structure_score(device)


def robust_structure_score(device: Any) -> RewardFunction:
    """Explicit Robust-U1 alias for the SSIM structure reward."""
    return structure_score(device)


def robust_tinyclip_score(device: Any) -> RewardFunction:
    """Explicit Robust-U1 alias for the TinyCLIP semantic reward."""
    return tinyclip_score(device)


def build_flow_grpo_reward_registry() -> dict[str, RewardBuilder]:
    """Return reward builders ready to merge into Flow-GRPO score_functions."""
    return {
        "restoration": restoration_score,
        "structure": structure_score,
        "robust_structure": robust_structure_score,
        "tinyclip": tinyclip_score,
        "robust_tinyclip": robust_tinyclip_score,
    }


def register_flow_grpo_rewards(
    score_functions: MutableMapping[str, RewardBuilder],
    *,
    override: bool = True,
) -> MutableMapping[str, RewardBuilder]:
    """Register Robust-U1 rewards into a Flow-GRPO score_functions mapping.

    Set override=False to keep existing Flow-GRPO implementations for names such
    as tinyclip and use robust_tinyclip / robust_structure in config.reward_fn.
    """
    for name, builder in build_flow_grpo_reward_registry().items():
        if override or name not in score_functions:
            score_functions[name] = builder
    return score_functions


def is_flow_grpo_reference_reward(score_name: str) -> bool:
    """Return True when the reward should receive ref_images in Flow-GRPO."""
    return score_name in FLOW_GRPO_REFERENCE_REWARD_NAMES


__all__ = [
    "FLOW_GRPO_REFERENCE_REWARD_NAMES",
    "build_flow_grpo_reward_registry",
    "is_flow_grpo_reference_reward",
    "register_flow_grpo_rewards",
    "restoration_score",
    "robust_structure_score",
    "robust_tinyclip_score",
]
