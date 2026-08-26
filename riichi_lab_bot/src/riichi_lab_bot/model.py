"""bot 侧复用训练代码的 V18 模型与编码常量。

注意：bot 的旧输入协议（history/snapshot）依赖在 V18 当前局面重构中已删除；
本模块只保留包级 import 兼容，完整迁移列入 V18 待迁移项。
"""

from riichi_ppo_v1.model.architecture import (
    KyokuTransformerActorCritic,
    ModelConfig,
)
from riichi_ppo_v1.model.encoding_protocol import (
    ENCODING_PROTOCOL_VERSION,
    QUERY_ROW_WIDTH,
    TOKEN_NUMERIC_WIDTH as NUMERIC_WIDTH,
    TOKEN_ROW_WIDTH as TOKEN_WIDTH,
)
from riichi_ppo_v1.model.schema import NUM_ACTIONS, TOKEN_SCHEMA_VERSION

__all__ = [
    "ENCODING_PROTOCOL_VERSION",
    "KyokuTransformerActorCritic",
    "ModelConfig",
    "NUM_ACTIONS",
    "NUMERIC_WIDTH",
    "QUERY_ROW_WIDTH",
    "TOKEN_SCHEMA_VERSION",
    "TOKEN_WIDTH",
]
