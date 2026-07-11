"""MJAI 九维事件序列 Transformer 模型包。

本包提供当前唯一 PPO 模型入口 ``build_model(model_size)``。
PPO 训练代码直接导入本包，不再通过模型工厂按名称动态加载。
"""

from model.mahjong_model.model import (
    NUM_ACTIONS,
    KyokuTransformerActorCritic,
    MahjongModelConfig,
    build_model,
    make_mahjong_model_config,
)

__all__ = [
    "NUM_ACTIONS",
    "KyokuTransformerActorCritic",
    "MahjongModelConfig",
    "build_model",
    "make_mahjong_model_config",
]
