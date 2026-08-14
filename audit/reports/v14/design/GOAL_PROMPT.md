# V14 PPO Goal 提示词（可直接复制）

> 本文件即 `/goal` 的输入。复制下面「推荐提示词」整段（以 `/goal` 开头）在新 Codex 会话中发送；
> 实现完成后训练、评测、日志与结论统一放在 `audit/reports/v14_ppo_20260812/`，
> checkpoint 放 `checkpoints/train_riichi_ppo_v14/`。

## 使用方法

1. 在仓库根目录 `/mnt/disk1/hubowen/zenith` 新开一个 Codex 会话（确保 `/goal` 可用；不可用先运行 `codex features enable goals`）。
2. 复制下方「推荐提示词」整段粘贴发送。
3. 用 `/goal` 查看状态，`/goal pause` / `/goal resume` / `/goal clear` 控制运行。
4. 中途看 `audit/reports/v14_ppo_20260812/PROGRESS.md`；结束后看 `EXPERIMENT_RESULTS.md`。

---

## 推荐提示词（复制这一段）

````text
/goal 按本提示词实现并运行 V14 PPO 训练：在不重训 SFT、不改 Actor 结构/Token Schema 13 的前提下，
从 checkpoints/train_riichi_v13_sft/best_heuristic.pt 初始化，给 Critic 增加“live wall 接下来 5 张牌”
的有序特权输入，40 update Critic Bootstrap 后进入 PPO Joint 训练，共 800 updates；
checkpoint 每 30 update 保存一次，每次保存后同步阻塞跑 1v3 1600 半庄评测（10 进程 × 160 半庄），
评测通过后训练才继续下一轮。

开始前先完整阅读：riichi_ppo_v1/docs/v13_sft.md、riichi_ppo_v1/model/architecture.py、
riichi_ppo_v1/model/critic_features.py、riichi_ppo_v1/model/bridge.py、riichi_ppo_v1/training/train.py、
riichi_ppo_v1/training/worker.py、riichi_ppo_v1/training/inference.py、riichi_ppo_v1/training/learner.py、
riichi_ppo_v1/sft/head_to_head_1v3.py、riichi_ppo_v1/configs/training.yaml、
riichi_ppo_v1/configs/goal/e2_opponent_mix.yaml，以及本目录 PROGRESS.md。

## 0. 背景与目标

V13 SFT 已提供强初始策略，且 best_heuristic.pt 的 model state 包含完整 critic 权重（已实测），
因此可以直接严格 load_state_dict 后“外挂”训练 Critic，不重新做 SFT。
V14 只增强 Critic 的 privileged information（对手隐藏手牌之外，新增未来 5 张 live wall 牌），
不改 Actor 与共享公共表征的输入契约。训练目标不是优化 loss，而是产出 V14 PPO checkpoint 与
对 V13 SFT 的固定 1v3 1600 半庄评测曲线（一位率、前二率、平均顺位、平均分差及 bootstrap CI）。

## 1. 总体路线与关键参数（决策已锁定，不要自行更改）

- 初始化：checkpoints/train_riichi_v13_sft/best_heuristic.pt（继承全部权重，value head 置零）。
- 阶段：update 1–40 Critic Bootstrap（actor/shared 冻结，critic_public_grad_scale=0.0，
  critic_bootstrap_learning_rate=0.00002）；update 41+ Joint PPO（critic_public_grad_scale=0.25）。
- Reward：纯小局点棒差，reward = clip((score_after - score_before)/1000, -24, +24)，
  仅 end_kyoku 时发放一次，GAE 向局内传播；不使用 GRP、dense efficiency、启发式对手。
- SFT-KL：sft_kl_coef_start=0.02 → sft_kl_coef_end=0.002（按 800 updates 线性退火）；
  PPO target_kl=0.02、ppo_clip=0.2、update_epochs=4，保持现配置，不调高。
- 对手分布：70% 四席全 current；20% 随机 1 席替换为冻结 V13 SFT（greedy）；
  10% 随机 1 席替换为 historical PPO（greedy，冻结）。
- 历史池：checkpoint_dir 下 checkpoint_<NNNNN>.pt（原子保存，只有完整文件），
  仅当 update>=60 且 (当前 update - checkpoint update)>=60 时入池，池内均匀随机；
  池空时把 70/20 重归一为 70/30（current/sft）继续训练。
- 评测：关闭训练内旧启发式评测（evaluation_enabled=false）；每 30 update 保存 checkpoint 后，
  同步阻塞运行 1v3：10 个进程 × 160 半庄 = 1600 半庄，CUDA_DEVICE=0 5 个、CUDA_DEVICE=2 5 个，
  model_a=<最新 checkpoint>，model_b=best_heuristic.pt，候选席按 i%4 轮转，greedy。
- GPU：测试、训练、评测统一使用 CUDA_DEVICE=0,2（物理 GPU0+GPU3），不用 CUDA=3（物理 GPU4）；
  双卡训练 learner_gpus=2；单卡 smoke 用 CUDA_DEVICE=0。

## 2. 硬约束

- 所有 Python/训练/评测/冒烟/测试命令必须通过 conda run -n Mahjong-AI（或显式激活该环境）执行。
- 固定 seed=1；game_mode=4p-red-half；长期训练 kyokus_per_worker=16；
  冒烟/性能测试 target_kl=0.0、update_epochs=4、kyokus_per_worker=1，跑 3 次，第 1 次热身，
  报告第 2–3 次性能与耗时。
- 允许修改：RiichiEnv/riichienv-python（仅新增 BatchedRiichiEnv.walls() 及必要测试）、
  riichi_ppo_v1/**、riichi_ppo_v1/configs/goal/v14_ppo.yaml、
  checkpoints/train_riichi_ppo_v14/**、audit/reports/v14_ppo_20260812/**。
- 禁止修改：模型 backbone/层数/宽度/Policy head、actor token 编码与 Token Schema 13、
  riichi_lab_bot/**、SFT 训练代码与数据集；不重训 SFT；不删除/覆盖旧 checkpoints、数据集与评测结果。
- 未来五张牌是 Critic-only privileged information：只能进入 critic_factors，
  严禁进入 actor 的 token_factors/token_numeric，也不得改 SFT public token sequence。
- 每次只改一个子系统并跑对应单测；全部单测 + smoke 通过后才允许启动长训。

## 3. 实现步骤（按顺序执行）

### 3.1 RiichiEnv 扩展：BatchedRiichiEnv.walls()

- 在 RiichiEnv/riichienv-python/src/env.rs 的 BatchedRiichiEnv 增加公开 getter walls()，
  返回 Vec<Vec<u32>>（每桌一行，内容为当前 state.wall.tiles 的全部剩余牌）。
- 语义（已实测）：wall.tiles 前段为 live wall，后 14 张为 dead wall；下一张正常摸牌 =
  tiles[0]，因此“接下来 5 张 live wall 牌” = tiles[0:5]（按顺序）。
- 重装扩展：先 conda activate Mahjong-AI，再运行 RiichiEnv/scripts/install_conda_extension.sh
  （或等价 maturin build --release + pip install --no-deps --force-reinstall）。
- 验收：python -c "from riichienv import BatchedRiichiEnv; e=BatchedRiichiEnv(1, seed=42); _=e.reset(); print(len(e.walls()), len(e.walls()[0]), e.walls()[0][:5])"
  应输出 1、83（初始 69 live + 14 dead）及前 5 个 tile id；并在 step 一次后确认 walls()[0][0] 前移。

### 3.2 Critic 未来五张牌编码（riichi_ppo_v1/model/critic_features.py）

- 新增常量：SEGMENT_CRITIC_FUTURE_WALL=5、TOKEN_KIND_FUTURE_WALL=6、FIELD_FUTURE_WALL=3、
  FUTURE_WALL_TILE_COUNT=5。
- 新增 encode_future_wall_tokens(wall) -> list[tuple[int,...]]：对 wall[:5] 按位置 i=1..5 生成
  (SEGMENT_CRITIC_FUTURE_WALL, TOKEN_KIND_FUTURE_WALL, FIELD_FUTURE_WALL, i, suit, rank, red, 1, 0, 1)；
  suit/rank/red 沿用 _tile_factors/_is_red（red 仅 {16,52,88}）；tile id 非法时抛错（不允许静默丢弃），
  牌数不足 5 时缺位不生成 token（保持 next_wall_1..5 的顺序语义）。
- encode_critic_features 增加可选参数 future_wall_tiles=()，在对手手牌 token 之后按顺序追加未来牌 token。
- 不新增模型参数：所有取值都在现有 TOKEN_CARDINALITIES 容量内（segment≤7、kind≤31、position≤7），
  因此 checkpoint 兼容性不受影响。

### 3.3 Worker / Bridge 接线

- worker.py：RolloutWorker 初始化时 self.walls = list(self.envs.walls())；每次 step_batch 与
  reset_indices 后立即刷新 self.walls；_submit_model_actions 调 bridge.prepare(decisions, analysis_batch,
  walls=self.walls)。
- bridge.py：BatchedStateBridge.prepare 增加可选参数 walls: list[list[int]] | None = None；
  critic_feature_encode 阶段对每个 decision 按 env_index 取 walls[env_index][:5] 编码并追加；
  walls 为 None（评测、lab_bot、旧调用）时行为与现在完全一致（不生成未来牌 token）。
- 关键顺序校验：同一 decision 的 critic token 顺序必须是 对手手牌 → next_wall_1..5 → value query。

### 3.4 语义校验（riichi_ppo_v1/model/semantic_validation.py）

- assert_critic_token_semantics 放行 segment=5 行，并校验：kind=6、field=3、slot3∈{1..5} 且不重复、
  marker=1、suit/rank 合法、red 仅在 rank=5 且 suit∈{1,2,3} 时可为 1。
- assert_actor_token_semantics 保持拒绝任何 critic-only segment（含 5），并在单测中显式断言
  actor factors 不含未来牌 token（无泄漏）。

### 3.5 对手分布与历史池

- worker.py set_rollout_context 扩展 opponent_mix：支持 current_frac/sft_frac/historical_frac/random_frac；
  按 roll 选择整桌类型（全 current / 1 席 sft / 1 席 historical / 1 席 random），historical 席标签为
  "history:<u%03d>"。只对 current 席记录 transition/reward（沿用现有逻辑）。
- 历史池枚举：扫描 checkpoint_dir 下 checkpoint_<5位数字>.pt，解析 update 号；
  池空时把 current_frac/sft_frac 重归一（70/30 相对比例）。
- inference.py：InferenceActor 增加 history_models: dict[str, nn.Module] 懒加载缓存；
  按 namespace "history:uXXX" 从 checkpoint 的 payload["model"] + model_config 构造
  KyokuTransformerActorCritic（eval、requires_grad_(False)），复用现有 _run_full_forward 路径（greedy）。
  加载失败或文件不存在时抛明确错误，不得静默降级。
- train.py 保存 checkpoint 改为原子写入：先写 checkpoint_XXXXX.pt.tmp，os.replace 为最终文件名
  （latest.pt 同理），避免历史池/评测读到半截文件。

### 3.6 训练配置（新增 riichi_ppo_v1/configs/goal/v14_ppo.yaml）

```yaml
# V14：Critic-only future 5 wall tiles；从 V13 SFT 初始化；不重训 SFT。
seed: 1
kyokus_per_worker: 1
iterations: 800
total_updates: 800
checkpoint_interval_updates: 30
checkpoint_dir: checkpoints/train_riichi_ppo_v14
init_model: checkpoints/train_riichi_v13_sft/best_heuristic.pt

critic_bootstrap_updates: 40
critic_bootstrap_learning_rate: 0.00002
critic_public_grad_scale: 0.25
zero_value_head_on_sft_init: true

kyoku_reward_clip_points: 24000
dense_efficiency_weight: 0.0

opponent_mix:
  enabled: true
  current_frac: 0.7
  sft_frac: 0.2
  historical_frac: 0.1
  random_frac: 0.0
  historical_min_update: 60
  historical_lag_updates: 60

ppo_clip: 0.2
target_kl: 0.02
update_epochs: 4
gamma: 0.99
gae_lambda: 0.95
value_coef: 0.5
value_loss: huber
value_target_normalization: batch_std
max_grad_norm: 0.5
entropy_start: 0.01
entropy_end: 0.001
sft_kl_coef_start: 0.02
sft_kl_coef_end: 0.002
sft_kl_anneal_updates: 800
learning_rate: 0.00002
actor_learning_rate: 0.00002
shared_learning_rate: 0.000005
critic_learning_rate: 0.00004
warmup_fraction: 0.02

evaluation_enabled: false
eval1v3_enabled: true
eval1v3_interval_updates: 30
eval1v3_processes: 10
eval1v3_hanchans_per_process: 160
eval1v3_parallel_hanchans: 160
eval1v3_model_b: checkpoints/train_riichi_v13_sft/best_heuristic.pt
eval1v3_seed_base: 20260812
eval1v3_devices: ["0", "2"]
eval1v3_output_dir: audit/reports/v14_ppo_20260812/eval
```

### 3.7 1v3 同步评测

- riichi_ppo_v1/sft/head_to_head_1v3.py：evaluate_1v3 的 model_a 结果中增加
  point_diff_samples（逐半庄 point_diff 列表，float），保持其余字段不变（向后兼容）。
- 新增 riichi_ppo_v1/sft/head_to_head_1v3_shards.py：
  run_sharded_1v3(checkpoint, model_b, *, update, processes=10, hanchans_per_process=160,
  parallel_hanchans=160, devices=("0","2"), seed_base, output_dir) -> dict：
  - 每个 shard 子进程用 [sys.executable, "-m", "riichi_ppo_v1.sft.head_to_head_1v3",
    --model-a ... --model-b ... --hanchans 160 --parallel-hanchans 160 --seed-base <base+shard>
    --device cuda --output <shard json>] 启动，前 5 个进程 env CUDA_DEVICE=0，后 5 个 env CUDA_DEVICE=2；
  - Popen 全部 10 个后统一 wait（同步阻塞），任一失败则整体报错并在 PROGRESS.md 记录；
  - merge_1v3_shards(shards) 汇总 1600 半庄口径：first_place/top2/fourth 计数与率、mean_rank、
    point_diff_mean、2000 次 pooled bootstrap 95% CI（合并所有 point_diff_samples），
    以及按 kyoku_count 加权的 model_a kyoku 指标（win/deal_in/tsumo_loss/riichi/draw_tenpai 等）；
  - 汇总写 output_dir/vs_sft_u<update>.json，shard 原文写 output_dir/shards/vs_sft_u<update>_shard<NN>.json。
- train.py：run_1v3_evaluation(update) 在 checkpoint 保存后同步调用 run_sharded_1v3；
  汇总 JSON 已存在则跳过；打印一行摘要（first_place_rate/top2_rate/mean_rank/point_diff_mean/CI）。
- 评测期间训练暂停（这是用户明确要求：不同时跑两个重任务，避免互相卡顿）。

## 4. 测试与验收（全部通过才启动长训）

新增/更新单测（pytest）：
- RiichiEnv/tests/env/test_batched_walls.py：walls() 数量与长度、tiles_left 一致、step 后头部前移、
  reset_indices 后刷新。
- riichi_ppo_v1/tests/unit/test_critic_features.py：受控墙编码 5 个有序未来牌 token；缺牌时只生成
  存在的 token；red5 标志；非法 tile 抛错；actor factors 不含未来牌/对手手牌 token。
- riichi_ppo_v1/tests/unit/test_semantic_validation.py：assert_critic_token_semantics 对新旧行都通过；
  位置重复/越界/kind 错误被拒绝。
- riichi_ppo_v1/tests/unit/test_worker_opponent_mix.py：池空回退 70/30；滞后 60 过滤；
  history:uXXX 标签格式；只记录 current 席。
- riichi_ppo_v1/tests/unit/test_head_to_head_1v3_shards.py：合成 10×160 shard 与直接 1600 口径一致
  （计数、加权指标、pooled CI）。
- 兼容性：用 best_heuristic.pt 对新模型 strict load_state_dict 成功（含 critic 权重）、value head 置零、
  现有 test_head_to_head.py / test_learner.py / lab_bot 测试不回归。

性能 smoke（CUDA_DEVICE=0，单卡）：
```bash
conda run --no-capture-output -n Mahjong-AI riichi-ppo-smoke --device cuda \
  --config riichi_ppo_v1/configs/goal/v14_ppo.yaml --kyokus 1
```
跑 3 次，报告第 2–3 次：sps、rollout_wall_s、update_wall_s、padding 指标（对比新增 5 个 critic token
后的变化）；若明显劣化（>10%）先定位再长训。

## 5. 长训启动与监控

全部测试通过后启动：
```bash
cd /mnt/disk1/hubowen/zenith
nohup env CUDA_DEVICE=0,2 conda run --no-capture-output -n Mahjong-AI \
  python -m riichi_ppo_v1.training.train --config riichi_ppo_v1/configs/goal/v14_ppo.yaml \
  > audit/reports/v14_ppo_20260812/train.log 2>&1 &
```
- 每 30 update：自动保存 checkpoint 并同步阻塞跑 1600 半庄 1v3，输出
  audit/reports/v14_ppo_20260812/eval/vs_sft_uXXX.json。
- 每 60 update 更新一次 PROGRESS.md（日期、update、训练指标摘要、最近 1v3 指标、异常、下一步）。
- 监控项：rollout reward、value_loss/EV、sft_reference_kl、actor/critic/shared grad_norm、
  history 对手使用计数（可在 rollout 日志/指标里加 opponent_mix 元数据）、评测耗时。

## 6. 交付物

- 代码改动（RiichiEnv walls()、critic 编码、worker/bridge/inference/train.py、评测 shard 工具）；
- riichi_ppo_v1/configs/goal/v14_ppo.yaml；checkpoints/train_riichi_ppo_v14/ 下 u30…u780 的 checkpoint；
- audit/reports/v14_ppo_20260812/train.log、eval/vs_sft_u*.json（含 shards/）、PROGRESS.md、
  EXPERIMENT_RESULTS.md（最终结论：1v3 曲线、最优 checkpoint、风险、下一步）；
- 如需 git 提交，用中文提交消息，只包含本 goal 相关改动。

## 7. 受阻停止条件

以下情况停止并报告（已尝试路径、证据、阻塞原因、下一步需要的输入）：
GPU/环境不可用、maturin 构建或 import 失败、任何必测单测连续 3 次针对性调整仍失败、
长训连续 3 次 update 出现环境级崩溃、评测工具跑不通、或出现未来牌语义泄漏且无法修复。
正常完成 800 updates 后写 EXPERIMENT_RESULTS.md 收尾。
````

## 一句话版本（先建立目标，再逐步细化）

```text
/goal 实现并运行 V14 PPO：Critic 增加未来 5 张 live wall 牌（Critic-only），从 V13 SFT 初始化，
40 update bootstrap 后 PPO Joint 800 updates；对手 70/20/10（current/SFT/history，池空 70/30）；
每 30 update 原子保存 checkpoint 并同步阻塞跑 1v3 1600 半庄（10×160，CUDA=0/2），
全部测试通过后启动长训，输出到 audit/reports/v14_ppo_20260812/ 与 checkpoints/train_riichi_ppo_v14/。
```
