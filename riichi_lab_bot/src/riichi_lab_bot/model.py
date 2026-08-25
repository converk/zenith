"""bot 侧复用训练代码的 V18 模型与编码常量。"""

from riichi_ppo_v1.model.architecture import (
    KyokuTransformerActorCritic,
    ModelConfig,
    NUM_ACTIONS,
    NUMERIC_WIDTH,
    TOKEN_CARDINALITIES,
    TOKEN_WIDTH,
)
from riichi_ppo_v1.model.encoding_protocol import (
    ENCODING_PROTOCOL_VERSION,
    QUERY_ROW_WIDTH,
    SNAPSHOT_FACTOR_WIDTH,
    SNAPSHOT_FIELD_COUNT,
    SNAPSHOT_NUMERIC_WIDTH,
)
from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION

__all__ = [
    "ENCODING_PROTOCOL_VERSION",
    "KyokuTransformerActorCritic",
    "ModelConfig",
    "NUM_ACTIONS",
    "NUMERIC_WIDTH",
    "QUERY_ROW_WIDTH",
    "SNAPSHOT_FACTOR_WIDTH",
    "SNAPSHOT_FIELD_COUNT",
    "SNAPSHOT_NUMERIC_WIDTH",
    "TOKEN_CARDINALITIES",
    "TOKEN_SCHEMA_VERSION",
    "TOKEN_WIDTH",
]
