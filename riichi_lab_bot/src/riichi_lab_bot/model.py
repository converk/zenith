"""Canonical V13 model exports shared with the training code path.

The bot deliberately re-exports the training-side architecture and schema
constants instead of maintaining a second model definition that can drift
from the checkpoint contract.
"""

from riichi_ppo_v1.model.architecture import (
    KyokuTransformerActorCritic,
    ModelConfig,
    NUM_ACTIONS,
    NUMERIC_WIDTH,
    TOKEN_CARDINALITIES,
    TOKEN_WIDTH,
)
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION

__all__ = [
    "KyokuTransformerActorCritic",
    "ModelConfig",
    "NUM_ACTIONS",
    "NUMERIC_WIDTH",
    "TOKEN_CARDINALITIES",
    "TOKEN_SCHEMA_VERSION",
    "TOKEN_WIDTH",
]
