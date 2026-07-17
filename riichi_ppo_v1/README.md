# RiichiEnv PPO 训练器

独立的四麻 PPO 训练器。策略分支输入是 Rust `MjaiKyokuStateMachineManager` 生成的公开语义
token，策略头输出固定 241 个动作槽位。value 分支在共享策略 hidden 的基础上，额外接收
critic-only 的三家对手手牌 token；可通过 `critic_include_public_state` 追加三家公开牌河和
副露的紧凑汇总。这些额外特征不进入策略分支。

## 目录

```text
riichi_ppo_v1/
├── model/       # Transformer 架构、RiichiEnv/MJAI/Rust 转换和协议验证
├── training/    # rollout、批量推理、PPO 优化、训练/验证命令入口
├── configs/     # 模型、环境、训练三类默认配置
├── docs/        # 动作空间与事件协议
└── tests/       # unit、integration、protocol 回归测试
```

## 训练集成与接口

```text
RiichiEnv Observation
  -> Python bridge（事件提取、快照、MJAI 合法动作）
  -> Rust 状态机（公开 history、当前状态后缀、241 维 mask）
  -> Transformer actor（追加 learned query，输出 policy logits）
       -> critic-only 对手手牌 token -> Transformer critic（输出 value）
  -> action_id -> 原始 MJAI JSON -> RiichiEnv Action -> env.step
```

对手手牌按非零牌种生成 token，牌数写在 token factor 的 `count` 字段中。启用
`critic_include_public_state` 后，牌河按对手、牌种与赤五状态聚合（不保留顺序和手切/摸切）；
每组副露保留类型、来源、组成牌种计数、赤五和组内索引。当前 critic 不接收额外 numeric 或 dense
牌种计数输入；红五在手牌 token 中折叠为普通五。

状态字段、可见性、token 编码和 Python/Rust 张量接口以
[docs/KyokuEventTupleProtocol.md](docs/KyokuEventTupleProtocol.md) 为准。固定动作槽位、红五区分和
MJAI/RiichiEnv 回转以 [docs/KyokuActionSpace.md](docs/KyokuActionSpace.md) 为准。状态机只将环境已广播
确认的事件写入 history；模型选择通过当前合法动作 mask 约束，不直接修改状态。

checkpoint 保存模型权重、优化器状态、模型配置、训练配置、迭代号和随机数状态。加载使用
PyTorch 严格模型权重校验；训练产物不提供格式迁移或兼容读取。

从仓库根目录先激活训练环境，并从源码安装两个扩展。`riichi` 使用仓库提供的安装脚本，脚本会把
PyO3/Maturin 明确绑定到当前 Conda 的 Python，避免扩展被装入系统 Python 或其他环境：

```bash
conda activate Mahjong-AI
bash riichi/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
```

默认配置随 Python 包一起安装，按职责拆分为：

- `configs/training.yaml`：训练/运行/模型/环境/优化器/reward/checkpoint 等会影响训练行为的参数；
- `configs/monitoring.yaml`：profiling、CUDA 同步诊断和 GPU telemetry 等只影响监控的参数。

`riichi-ppo-train` 会按上述顺序合并默认值。`--training-config` 可覆盖训练参数；
`--model-config`、`--environment-config` 作为兼容入口仍保留；原有 `--config` 保留为最后合并的完整 YAML overlay。默认
`checkpoint_dir` 仍为 `checkpoints/riichi_ppo_v1`，以保留本地测试产物的位置。

单卡训练与 GPU 测试默认固定到物理 0 号卡：

```bash
CUDA_DEVICE=0 riichi-ppo-smoke --device cuda
```

训练入口会将项目约定的 `CUDA_DEVICE=0` 转成 CUDA 标准的
`CUDA_VISIBLE_DEVICES=0`。此时进程内可见的卡编号是 `cuda:0`，因此不要同时传
`--device cuda:0` 以外的物理卡编号。双卡训练使用 `CUDA_DEVICE=0,3`，其中 `3`
对应物理 GPU 4，并传 `--learner-gpus 2`。只跑 Rust 单测时不需要 CUDA；在 Conda
环境下请补充动态库路径：

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" cargo test --manifest-path riichi/Cargo.toml
```

训练：

```bash
riichi-ppo-train
```

最小端到端检查（一个 Ray worker、一个环境、一个小局和一次更新）：

```bash
riichi-ppo-smoke --device cpu
```

## 模型—环境语义转换验证

固定回归用例覆盖 241 个动作槽位、红五/手切摸切、所有四麻动作窗口，以及事件到语义 token
的字段和生命周期边界。还可以运行真实环境的种子回放：它会从 241 维 mask 选择 action id、
经 bridge 解码后实际提交给环境，而不是直接随机选择环境动作。

```bash
conda run -n Mahjong-AI riichi-ppo-validate --games 128 --output riichi_ppo_v1_coverage.json
```

输出 JSON 分别列出当前窗口中提供的动作和实际提交的动作（类型与 ID）、观察到的环境事件以及
自然回放未命中的项目。未命中只表示该随机样本未覆盖；MJAI、mask、解码动作或环境选择出现语义
不一致时，命令会立即失败。

训练时，每张桌在一个半庄开始时随机、不重复地抽取两名训练座位，并在该半庄内保持不变。
四个座位的每一个合法决策都由同一份当前 rollout 模型产生；只有被抽样的两席会保存
transition 并参与 PPO update。半庄结束后才会为该桌重新抽取两席。

`kyokus_per_worker` 是开始 drain 前的最少完成小局数。达到该值后，worker 会冻结已经
结束当前小局的桌子，只推进其他尚未结束的小局直至全部结算；因此不会因 PPO collection
边界丢弃已采样但尚未得到终局 reward 的行为。实际完成小局数可能高于该下限。

训练 worker 使用原生 `BatchedRiichiEnv`：每个 tick 会同步收集 worker 内全部环境的四家
observation 增量、构造状态机输入，并在释放 GIL 后并行推进环境。默认配置为 12 个 Ray
worker、每 worker 32 张桌、每 worker 4 条原生环境线程。默认 `learner_gpus: 2` 使用
`CUDA_DEVICE=0,3` 启动两个 GPU actor，worker 按 rank 近似平均分配；每个 actor 只服务自己那部分
worker 的 rollout inference。PPO update 使用 PyTorch DistributedDataParallel：两个 rank 对相同 global
minibatch 取不同 shard 分别反传，梯度经 NCCL all-reduce 后同步更新。worker 本身不创建 CUDA
context。
可通过 `envs_per_worker` 与 `env_step_threads` 调整。

在 CUDA 支持 BF16 时，rollout inference 和 PPO update 的模型前向均使用 BF16 autocast；
policy/value 输出、PPO loss、梯度、模型权重与 AdamW 状态保持 FP32。CPU 或不支持 BF16 的
CUDA 设备则全程使用 FP32，不会回退到 FP16。

## 性能监控

默认启用逐阶段 profiling。每轮会把完整指标写入 `checkpoint_dir/performance.jsonl`，并在
`checkpoint_dir/tensorboard` 中记录同名标量；控制台会逐行打印 rollout worker、rollout
inference actor 和 PPO update 的阶段表。worker 阶段同时给出 4 个 worker 的 total 时间
mean/max/min 与调用数；inference actor 的时间按 iteration 汇总一次，不会因 RPC 合批而重复计数。
重点指标包括：

- rollout：observation/合法动作扫描、事件提取、Rust 事件更新、MJAI JSON、snapshot、
  Rust 输入构造、GPU inference RPC、queue wait、跨 worker host collation、H2D、
  full-forward、低频 CUDA event、采样、MJAI 回转、原生批量 step、GAE 和 reset；
- update：长度分桶、每 minibatch padding/collate/H2D、优势归一化、前向、loss、反向、裁剪、optimizer；
- GPU：利用率、显存控制器利用率、显存用量、PyTorch allocated/reserved/peak、功耗、温度、SM 时钟。

`iteration/algorithm_wall_s` 从本轮 rollout 准备开始，到 PPO actor 完成 update 为止；它包含
rollout、profile RPC、主进程 transition 拼接和传输、PPO update，但不包含日志、文件、checkpoint
或 eval。主指标 `iteration/sps` 是采样席决策数除以该时间；
`iteration/model_forward_sps` 是四席总模型推理负载，`effective_transitions_per_s` 则是最终
进入 PPO update 的 transition 吞吐。

默认的 `profile_cuda_sync: false` 不会为每个 GPU 阶段加同步栅栏。每
`profile_cuda_event_interval` 次 rollout forward 会使用一次 CUDA event 采样准确 GPU 时长；
需要逐阶段诊断时再临时设为 `true`。`gpu_sample_interval_s` 控制后台 `nvidia-smi`
采样周期，默认 0.25 秒。

### 训练质量监控

除性能指标外，训练还会写入 `checkpoint_dir/metrics.jsonl` 与 TensorBoard 语义指标。JSONL 的每行
包含 schema 版本、update、累计 PPO 决策数、累计小局数、来源（`train` 或 `evaluation`）和有限标量；
恢复训练时累计计数会从该文件续接。日志仅包含公开 MJAI 结算事件、动作类别和分数，绝不包含暗手、墙牌、
随机状态或 critic-only 输入。

`train/kyoku/*`、`train/match/*`、`train/action/*`、`train/efficiency/*` 与
`train/defense/*` 用于诊断在线 rollout；`ppo/explained_variance`、`ppo/buffer/*`、
`ppo/ratio` 与 `ppo/ratio_p95` 用于诊断 value 拟合和 PPO policy drift。每 25 个 update 写入
return、advantage、value、合法动作数和 token 长度直方图。

默认还会在 update 0、每 25 个 update 和结束时运行固定基线评测。它使用 12 个确定 seed、候选策略
4 个座位轮换，共 48 个半庄；候选模型贪心决策，对手为固定的效率/防守启发式混合。应以
`eval/match/rank_mean`（越低越好）和 `eval/match/first_rate`（越高越好）判断跨课程阶段的真实进步，
而不是把变化中的自博弈 reward 当作实力指标。相关开关和预算位于 `configs/monitoring.yaml`。

运行一轮目标负载 benchmark 而不修改默认长期训练配置：

```bash
CUDA_DEVICE=0 conda run -n Mahjong-AI riichi-ppo-train \
  --device cuda --iterations 1 \
  --num-workers 4 --envs-per-worker 48 --kyokus-per-worker 1 \
  --update-epochs 4 --minibatch-size 768 --target-kl 0.0 \
  --checkpoint-dir checkpoints/riichi_ppo_v1_benchmark_4w48e
```

将 `--num-workers 4` 改为 `1` 即可运行 1×48。吞吐 benchmark 保持默认
`--no-profile-cuda-sync`；先用 `nvidia-smi -i 0` 确认物理第 1 张卡没有外部
compute process，再分别热身一次、重新启动进程并记录三次结果。比较
`rollout/wall_s`、`iteration/model_forward_sps` 与
`rollout/inference_actor/inference/full_forward_rows_mean`。

双卡一轮 benchmark：

```bash
CUDA_DEVICE=0,3 conda run -n Mahjong-AI riichi-ppo-train \
  --device cuda --learner-gpus 2 --iterations 1 \
  --num-workers 4 --envs-per-worker 48 --kyokus-per-worker 1 \
  --update-epochs 4 --minibatch-size 768 --target-kl 0.0 \
  --checkpoint-dir checkpoints/riichi_ppo_v1_benchmark_2gpu_4w48e
```
