# RiichiEnv SFT → PPO 训练器

该目录只包含当前训练流程：从天凤牌谱准备 V16 SFT 数据、训练策略模型、以
V16 SFT/GRP 产物初始化 V17 PPO，并执行固定 1v3 对抗评测。

## 安装

```bash
conda activate Mahjong-AI
bash RiichiEnv/riichienv-state-machine/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
```

## V17 流程(2026-08-19,Mortal 式 GRP 纯奖励)

V17 在 V16-small 拓扑基础上引入 Mortal 式 GRP(全局排名预测)作为唯一奖励
信号:

- GRP 输入为每个 StartKyoku 的 7 维全局状态
  `[grand_kyoku, honba, kyotaku, s0/1e4, s1/1e4, s2/1e4, s3/1e4]`;每半庄所有
  prefix 预测最终四人排名的 24 类全排列。
- 模型:`7 → 2 层 GRU(hidden=64) → concat → Linear(128,128) → ReLU →
  Linear(128,24)`;离线训练 `batch_size=512`、AdamW、`lr=1e-5`,保存 validation
  loss 最低的 checkpoint(`checkpoints/train_riichi_v17/grp/best.pt`),
  训练完成后完全冻结。
- PPO 奖励 = 纯 GRP delta(排名 utility `[1, 1/3, -1/3, -1]`),删除小局点差
  分量;每 update 收集 2048 个完整半庄,双卡 DDP(每卡 minibatch 1536 →
  global effective 3072,`update_epochs=1`,显存不足可用 gradient accumulation
  兜底);对手只使用 current self-play。
- 每 5 updates 保存 checkpoint 并对 V16 SFT 1v3 评测 4000 半庄(10 进程 ×
  400,双卡各 5 进程,与训练同卡串行,分片内部按 200 半庄一批分批推进);训练
  结束后按 1V3 表现选择最佳 checkpoint。配置:
  `v17_grp.yaml` 与 `v17_ppo.yaml`,产物落 `train_riichi_v17/`、
  `logs/v17/`、`audit/reports/v17/`。

## V16-small 当前流程(2026-08-17)

V16 版本命名保持不变,仅把隐藏层调整为 V16-small(总参数约 3.0M):
`d_model=192`、Q/KV=12/3、head_dim=16、FFN=576、3 Shared + 1 Actor +
2 Critic。输入/输出协议不变,既有 V16 编码数据可直接复用。

SFT 数据从 40% 扩到 60%:复用
`datasets/tenhou_sft_2024_2025_encoded_40pct_v16`,仅追加 remainder=2 的
20% 新编码到 `datasets/tenhou_sft_2024_2025_encoded_60pct_v16`;GRP 模型与
奖励契约不修改。

## 当前训练流程

准备数据并训练一轮 actor-only SFT：

```bash
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-prepare \
  --archive-dir datasets/raw/tenhou_2024_2025 \
  --output datasets/tenhou_sft_2024_2025
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-precompute \
  --source datasets/tenhou_sft_2024_2025 \
  --output datasets/tenhou_sft_2024_2025_encoded_60pct_v16 \
  --subset-denominator 5 --subset-remainders 0,1,2 --workers 16
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-sft-train \
  --config riichi_ppo_v1/configs/v16_sft.yaml
```

编码数据使用 fp16 numeric、bit-packed legal mask 和可变长三段输入；manifest 绑定
V16 encoding protocol，并验证公开牌河与有序 MJAI history 一致，协议变化时会拒绝
旧缓存。

随后以 V16 SFT checkpoint 与冻结 GRP checkpoint 启动 V17 PPO：

```bash
RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 \
  conda run -n Mahjong-AI python -m riichi_ppo_v1.training.train \
  --config riichi_ppo_v1/configs/v17_ppo.yaml --device cuda --learner-gpus 2
```

actor-only SFT checkpoint 会同时成为初始策略和冻结的 KL reference。V17 PPO 使用
Mortal 式 GRP delta 作为奖励,每 update 收集完整半庄,对手只使用 current
self-play。PPO checkpoint 保存冻结 reference 与逐 rank RNG,`resume` 可以精确延续
训练语义。

`learner_gpus` 同时决定 PPO update 的并行方式:`1` 时 driver 进程在单卡完成
update;`2` 时 driver 拉起两个常驻 learner 进程(各持一份模型),经 NCCL 做
双卡 DDP——rollout 在两个 rank 间轮询分片并补齐到一致的本地 minibatch 数,
advantage/returns 在完整 rollout 上先算好再随分片下发,每步梯度跨卡平均后
同步执行优化器 step。两个推理 actor 仍各占一张卡,每次 update 后由 rank 0
把最新权重推送给它们。

`resume` 仅恢复当前 PPO checkpoint（包含 `ppo_format_version: 3`）。新
checkpoint 在 `extra_state.rank_rng_states` 中保存逐 rank RNG,双卡
`resume` 会精确恢复每个 rank 的训练随机状态。

PPO 唯一评测机制是固定 1v3 对抗:每 5 个 update 一次、10 个进程各 400 半庄
(共 4000 半庄,双卡各 5 进程),候选策略对阵由 `eval1v3_model_b` 指定的
对手模型。对手模型、种子基数、设备与输出目录都来自版本配置;结果写入 `eval1v3_output_dir`
(固定为 `audit/reports/<版本号>/eval`),摘要追加到该目录下的
`eval1v3.jsonl`,进度记录写入 `audit/reports/<版本号>/report/PROGRESS.md`。
训练结束后可用 `riichi_ppo_v1/evaluation/select_best_checkpoint` 按 1V3 表现
选择最佳 checkpoint。默认配置不含任何版本化对手或目录。

SFT 的固定启发式评测独立于 PPO reward，使用轮换座位的效率/防守启发式对手。它记录名次、分差、和牌、放铳、被自摸、流局、立直、副露及座位/阶段分组指标。

## 配置与验证

- `configs/sft.yaml`：V16 SFT 数据加载、优化器与 checkpoint 默认值。
- `configs/training.yaml`：V16/V17 中性 PPO 拓扑、优化器与 checkpoint 默认值。
- `configs/monitoring.yaml`：性能和公开语义指标采集，不改变训练算法。

## 产物目录规范

- 所有运行日志(json/txt/log)写入 `logs/<版本号>/`,禁止在 `logs/` 根目录或
  他处单独生成日志文件;运行脚本用 `tee` 把训练与 Ray/子进程输出重定向到该目录。
- 初始设计文档、实验报告、测试与验证脚本唯一存放于
  `audit/reports/<版本号>/` 的 `design/`、`report/`、`scripts/`,评测与验证输出
  放 `eval/`;这些固定类型子目录中的 design/report/scripts 进版本控制。
- checkpoint 固定保存在 `checkpoints/train_riichi_<版本号>/<阶段>`;现行数据集为
  `datasets/tenhou_sft_2024_2025`、
  `datasets/tenhou_sft_2024_2025_encoded_60pct_v16` 与
  `datasets/tenhou_grp_2024_2025_v17`。

最小 CPU 检查：

```bash
conda run -n Mahjong-AI riichi-ppo-smoke --device cpu
```

性能基准固定运行三轮；第 1 轮作为 warm-up，单独报告第 2–3 轮：

```bash
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-ppo-train \
  --device cuda --learner-gpus 2 --iterations 3 \
  --num-workers 12 --envs-per-worker 32 --kyokus-per-worker 16 \
  --update-epochs 4 --minibatch-size 512 --target-kl 0.0 \
  --checkpoint-dir checkpoints/train_riichi_benchmark
```

协议和动作空间仍以 [KyokuEventTupleProtocol.md](docs/KyokuEventTupleProtocol.md) 与 [KyokuActionSpace.md](docs/KyokuActionSpace.md) 为准。
