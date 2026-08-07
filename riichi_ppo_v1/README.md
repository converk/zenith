# RiichiEnv SFT → PPO 训练器

该目录只包含当前训练流程：从天凤牌谱准备 SFT 数据、训练策略模型、以 SFT 权重初始化四席自博弈 PPO，以及独立的固定启发式对局评测。

## 安装

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichi/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
```

## 当前训练流程

准备数据并训练一轮 actor-only SFT：

```bash
CUDA_DEVICE=0,3 conda run -n Mahjong-AI riichi-sft-prepare \
  --output datasets/tenhou_sft_2024_2025
CUDA_DEVICE=0,3 conda run -n Mahjong-AI riichi-sft-train \
  --dataset datasets/tenhou_sft_2024_2025
```

如只使用稳定抽样的 1/10 数据并消除训练期间的在线回放，可先在后台物化 actor-only 缓存：

```bash
nohup conda run -n Mahjong-AI riichi-sft-precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_2024_2025_encoded_10pct_v2 \
  --subset-denominator 10 --subset-remainder 0 --workers 8 \
  > logs/sft-precompute-10pct.log 2>&1 &
```

完成后直接把该目录传给 `riichi-sft-train --dataset`。编码数据使用 fp16 numeric、bit-packed legal mask 和可变长 token；其 manifest 绑定 token schema，并验证公开牌河与有序 MJAI history 一致，协议变化时会拒绝旧缓存。

随后以 SFT checkpoint 启动 PPO。PPO 永远由四席 current policy 自博弈；每个决策只在所属小局结算时获得该席的终局分差（`clip(delta_score / 1000, -12, 12)`）作为 reward。

actor-only SFT checkpoint 会同时成为初始策略和冻结的 KL reference。导入时 PPO 将新 value head 置零，前 30 个 update 固定 actor 与共享 public backbone，只训练 critic 私有分支；之后进入标准联合 PPO，并将当前策略相对冻结 SFT reference 的完整合法动作分布 KL 加入 loss。KL 系数在长期训练中从 `0.02` 线性降至 `0.002`，因此 reference anchor 不会在后期完全消失。PPO checkpoint 会保存冻结 reference，`resume` 可以精确延续该语义。

当前 mid 网络使用 `3 层 shared public + 1 层 actor + 2 层 critic`。正式 PPO 对三个参数组分别使用 `2e-5 / 5e-6 / 4e-5` 的 actor/shared/critic 基础学习率，并使用相同的 warmup/衰减进度。value loss 进入 shared public backbone 的梯度额外乘以 `0.25`；policy gradient 不受该倍率影响。critic bootstrap 期间该倍率为 `0`。

两层 critic 会改变 checkpoint 参数形状。应使用当前代码重新训练 SFT，再以生成的新 checkpoint 初始化 PPO；早期单层 critic checkpoint 不能通过严格的完整模型加载继续使用。

```bash
CUDA_DEVICE=0,3 conda run -n Mahjong-AI riichi-ppo-train \
  --device cuda --learner-gpus 2 \
  --init-model checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

`resume` 仅恢复新格式 PPO checkpoint（包含 `ppo_format_version: 2`）。旧 PPO checkpoint 不能恢复优化器或旧课程状态，但可通过 `--init-model` 仅加载模型权重。

PPO 每 15 个 update 使用同一组固定种子执行一次座位均衡的启发式基线评测。
默认每次评测 96 个半庄；双 learner GPU 将其分成连续的 48 + 48 场并发
执行，每张卡最多并行 48 桌。候选策略使用贪心动作，对手在效率与防守
启发式策略之间轮换。评测曲线写入 `<checkpoint_dir>/tensorboard/`，
完整结果同时写入 `<checkpoint_dir>/evaluation.jsonl` 和 `metrics.jsonl`。

SFT 的固定启发式评测独立于 PPO reward，使用轮换座位的效率/防守启发式对手。它记录名次、分差、和牌、放铳、被自摸、流局、立直、副露及座位/阶段分组指标。

## 配置与验证

- `configs/sft.yaml`：SFT 数据加载、优化器、checkpoint 与启发式评测。
- `configs/training.yaml`：当前标准 PPO 的拓扑、优化器与 checkpoint。
- `configs/monitoring.yaml`：性能和公开语义指标采集，不改变训练算法。

最小 CPU 检查：

```bash
conda run -n Mahjong-AI riichi-ppo-smoke --device cpu
```

性能基准固定运行三轮；第 1 轮作为 warm-up，单独报告第 2–3 轮：

```bash
CUDA_DEVICE=0,3 conda run -n Mahjong-AI riichi-ppo-train \
  --device cuda --learner-gpus 2 --iterations 3 \
  --num-workers 12 --envs-per-worker 32 --kyokus-per-worker 1 \
  --update-epochs 4 --minibatch-size 512 --target-kl 0.0 \
  --checkpoint-dir checkpoints/train_riichi_benchmark
```

协议和动作空间仍以 [KyokuEventTupleProtocol.md](docs/KyokuEventTupleProtocol.md) 与 [KyokuActionSpace.md](docs/KyokuActionSpace.md) 为准。
