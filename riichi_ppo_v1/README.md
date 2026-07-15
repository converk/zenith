# RiichiEnv PPO v1

独立的四麻 PPO 训练器。模型输入是 Rust `MjaiKyokuStateMachineManager` 生成的
`KyokuEventTuple V3`，动作空间是 `KyokuActionSpace V2` 的 241 个动作。

从仓库根目录先激活训练环境，并从源码安装两个扩展。`riichi` 使用仓库提供的安装脚本，脚本会把
PyO3/Maturin 明确绑定到当前 Conda 的 Python，避免扩展被装入系统 Python 或其他环境：

```bash
conda activate Mahjong-AI
bash riichi/scripts/install_conda_extension.sh
bash RiichiEnv/scripts/install_conda_extension.sh
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
```

训练与 GPU 测试建议固定到物理 3 号卡：

```bash
CUDA_DEVICE=3 riichi-ppo-smoke --device cuda
```

训练入口会将项目约定的 `CUDA_DEVICE=3` 转成 CUDA 标准的
`CUDA_VISIBLE_DEVICES=3`。此时进程内可见的卡编号是 `cuda:0`，因此不要同时传
`--device cuda:3`。只跑 Rust 单测时不需要 CUDA；在 Conda 环境下请补充动态库路径：

```bash
LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" cargo test --manifest-path riichi/Cargo.toml
```

训练：

```bash
riichi-ppo-train --config configs/default.yaml
```

最小端到端检查（一个 Ray worker、一个环境、一个小局和一次更新）：

```bash
riichi-ppo-smoke --device cpu
```

训练时，每张桌在一个半庄开始时随机、不重复地抽取两名训练座位，并在该半庄内保持不变。
四个座位的每一个合法决策都由同一份当前 rollout 模型产生；只有被抽样的两席会保存
transition 并参与 PPO update。半庄结束后才会为该桌重新抽取两席。

`kyokus_per_worker` 是开始 drain 前的最少完成小局数。达到该值后，worker 会冻结已经
结束当前小局的桌子，只推进其他尚未结束的小局直至全部结算；因此不会因 PPO collection
边界丢弃已采样但尚未得到终局 reward 的行为。实际完成小局数可能高于该下限。

训练 worker 使用原生 `BatchedRiichiEnv`：每个 tick 会同步收集 worker 内全部环境的四家
observation 增量、构造状态机输入，并在释放 GIL 后并行推进环境。默认配置为 4 个 Ray
worker、每 worker 48 张桌、每 worker 4 条原生环境线程。唯一的 GPU PPO actor 同时持有
模型、optimizer 和按 `worker/env/player` 隔离的事件历史 KV cache：它串行执行 rollout inference
与 PPO update，因此没有第二份 GPU 模型或每轮权重同步；worker 本身不创建 CUDA context。
可通过 `envs_per_worker` 与 `env_step_threads` 调整。

## 性能监控

默认启用逐阶段 profiling。每轮会把完整指标写入 `checkpoint_dir/performance.jsonl`，并在
`checkpoint_dir/tensorboard` 中记录同名标量；控制台会逐行打印 rollout worker、rollout
inference actor 和 PPO update 的阶段表。worker 阶段同时给出 4 个 worker 的 total 时间
mean/max/min 与调用数；inference actor 的时间按 iteration 汇总一次，不会因 RPC 合批而重复计数。
重点指标包括：

- rollout：observation/合法动作扫描、事件提取、Rust 事件更新、MJAI JSON、snapshot、
  Rust 输入构造、GPU inference RPC、H2D、KV prefill/append、cache hit/miss、采样、
  MJAI 回转、原生批量 step、GAE 和 reset；
- update：padding/collate/H2D、优势归一化、索引、前向、loss、反向、裁剪、optimizer；
- GPU：利用率、显存控制器利用率、显存用量、PyTorch allocated/reserved/peak、功耗、温度、SM 时钟。

`iteration/algorithm_wall_s` 从本轮 KV cache 准备开始，到 PPO actor 完成 update 为止；它包含
rollout、profile RPC、主进程 transition 拼接和传输、PPO update，但不包含日志、文件、checkpoint
或 eval。主指标 `iteration/sps` 是采样席决策数除以该时间；
`iteration/model_forward_sps` 是四席总模型推理负载，`effective_transitions_per_s` 则是最终
进入 PPO update 的 transition 吞吐。

`profile_cuda_sync: true` 会在 learner 的各 GPU 阶段前后同步，得到可信的逐操作耗时，
但会降低训练吞吐；确定瓶颈后可改为 `false`。`gpu_sample_interval_s` 控制后台
`nvidia-smi` 采样周期，默认 0.25 秒。

运行一轮目标负载 benchmark 而不修改默认长期训练配置：

```bash
CUDA_DEVICE=3 conda run -n Mahjong-AI riichi-ppo-train \
  --config riichi_ppo_v1/configs/default.yaml --device cuda --iterations 1 \
  --checkpoint-dir checkpoints/riichi_ppo_v1_profile_4w48e_selfplay
```
