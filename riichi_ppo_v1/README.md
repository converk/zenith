# V19 SFT / V18 PPO 训练框架

本包的活跃 SFT 协议为 V19：**决策时刻状态快照 + 信念五头监督**（Shared 公共前缀 +
三家 Opponent Analysis + 信念 token + 每个合法动作一对 Offense/Defense Query，
全 token RoPE、公共双向 GQA、结构化 Actor mask）。V16/V17 配置与产物保留为冷存储，
不能由活跃 checkpoint、SFT、评测或 bot 路径加载；PPO/rollout 与 bot 的旧输入引用
为后续待迁移项。

## 安装与验证

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
python -m riichi_ppo_v1.tools.validate --parameter-contract
python -m pytest riichi_ppo_v1/tests
```

V19 SFT 参数拓扑为 `d_model=256`、16 Q heads、4 KV heads、`head_dim=16`、
`ffn_dim=704`、3 Shared + 2 Actor + 1 Critic，密集槽位 `dense_slot_dim=32`、
`dense_fusion_dim=512`，`context_tokens=320`，并含信念网络（五头 + 三家各 10 token）。
模型不包含 Q scorer、candidate-Q 输出、MHA 双分支或旧协议兼容 key。

## V19 SFT-ready 路径

唯一现行 SFT 自包含配置是 `configs/v19_sft.yaml`（V19 标准 SFT，
`datasets/tenhou_sft_2024_2025_encoded_60pct_v19` 已预处理完成，不再生成数据）。
V19 SFT 目标为 Actor BC 与信念五头监督联合
（`L_BC + belief_sft_coef·Σλ_k·L_k + λ_c·L_wait_danger`）：

```bash
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-train \
  --config riichi_ppo_v1/configs/v19_sft.yaml
```

训练可一键执行（产物地址与校验见脚本头部注释）：

```bash
bash audit/reports/v19/scripts/run_v19_precompute_and_sft.sh --skip-precompute
```

预计算 manifest 必须声明 `riichi-sft-encoded-v19`、protocol 19、冻结的 V19
contract hash、`belief_labels=true` 与 `belief_shape`；旧缓存会 fail closed。
actor-only BC + 信念只优化 Actor 参数（含 `belief_network`），并只保存可被 V19
精确加载的 Actor artifact。固定验证/checkpoint 节奏为每 3000 steps，最终评估为
96 半庄。

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

产物目录固定为 `checkpoints/train_riichi_v19/<阶段>`、`logs/v19/` 与
`audit/reports/v19/{design,eval,report,scripts}`。详细协议见
[V19 输入协议](docs/v19_input_protocol.md)，SFT 参数与生命周期见
[V19 SFT](docs/v19_sft.md)。
