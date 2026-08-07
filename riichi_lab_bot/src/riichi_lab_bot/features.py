"""Actor-visible public river and meld summary.

The encoding itself is delegated to the training-side
``riichi_ppo_v1.model.critic_features`` implementation so the bot and the
training path can never drift.  Only the single-observation adapter lives
here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from riichi_ppo_v1.model.critic_features import (
    collect_actor_public_table_state,
)
from riichi_ppo_v1.model.critic_features import (
    encode_public_summary as _encode_training_public_summary,
)


def encode_public_summary(
    observation: Any, observer: int
) -> np.ndarray:
    table = collect_actor_public_table_state(observation)
    return _encode_training_public_summary(table, observer).factors
