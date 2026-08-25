# V18 SFT/PPO 训练框架

本包的活跃协议仅为 V18：Objective Facts + 29 个 Atomic Snapshot 字段 + 每个
合法动作一对 Offense/Defense Query。V16/V17 配置与产物保留为冷存储，不能由
活跃 checkpoint、SFT、评测或 bot 路径加载。

## 安装与验证

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
python -m riichi_ppo_v1.tools.validate --parameter-contract
python -m pytest riichi_ppo_v1/tests
```

参数契约固定为 `d_model=256`、16 Q heads、4 KV heads、`head_dim=16`、
`ffn_dim=704`、3 Shared + 1 Actor + 2 Critic，完整 Actor-Critic 参数量必须位于
4.9M–5.1M。模型不包含 Q scorer、candidate-Q 输出或兼容 key。

## V18 SFT-ready 路径

现行自包含配置是 `configs/v18_sft.yaml`。本次升级只交付可训练接口，不生成完整
V18 数据集，也不启动正式 SFT/PPO：

```bash
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_2024_2025_encoded_60pct_v18 \
  --subset-denominator 5 --subset-remainders 0,1,2 --workers 16

CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-train \
  --config riichi_ppo_v1/configs/v18_sft.yaml
```

预计算 manifest 必须声明 `riichi-sft-encoded-v18`、protocol 18 和冻结的 V18
contract hash；旧缓存会 fail closed。actor-only BC 只优化 Actor 参数，并只保存
可被 V18 精确加载的 Actor artifact。固定验证/checkpoint 节奏为每 3000 steps，
最终评估为 96 半庄。

## PPO 与评测边界

V18 PPO 接口已消费新的张量与 checkpoint format 4，但本升级不运行 PPO。正式训练
仍须使用前台双卡启动、日志写入 `logs/<版本号>/`，并遵守唯一 1v3 评测机制：每
5 updates、10 进程 × 400 半庄、双卡各 5 进程；对手、种子、设备和输出目录只能
由自包含版本配置提供。

```bash
RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 \
  conda run -n Mahjong-AI python -m riichi_ppo_v1.training.train \
  --config <V18-自包含-PPO-配置> --device cuda --learner-gpus 2
```

产物目录固定为 `checkpoints/train_riichi_v18/<阶段>`、`logs/v18/` 与
`audit/reports/v18/{design,eval,report,scripts}`。详细协议见
[V18 输入协议](docs/v18_input_protocol.md)，SFT 参数与生命周期见
[V18 SFT](docs/v18_sft.md)。
