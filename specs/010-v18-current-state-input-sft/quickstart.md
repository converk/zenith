# Quickstart: V18 当前局面输入与 Actor 决策架构重构

## 环境

```bash
source /mnt/disk1/hubowen/miniconda3/etc/profile.d/conda.sh
conda activate Mahjong-AI
cd /mnt/disk1/hubowen/zenith
```

## 构建 Rust/PyO3 扩展（每次 Rust 改动后）

```bash
bash RiichiEnv/scripts/install_conda_extension.sh
```

校验：`python -c "import riichi, riichienv; print(riichi.ENCODING_PROTOCOL_VERSION, riichi.ANALYSIS_VERSION)"`
（期望 `18 4`），且 `riichienv.prepare_current_state_batch` 存在。

## 1. 协议/单测（无 GPU）

```bash
python -m pytest riichi_ppo_v1/tests/unit/test_v18_encoding_protocol.py \
  riichi_ppo_v1/tests/unit/test_v18_architecture.py \
  riichi_ppo_v1/tests/unit/test_v18_parameter_count.py \
  riichi_ppo_v1/tests/unit/test_v18_dense_embedding.py \
  riichi_ppo_v1/tests/unit/test_v18_sft_contract.py -q
```

预期：全部通过；参数报告 total < 6.0M，无 forbidden keys。

## 2. 真实 replay 编码一致性

```bash
python -m pytest riichi_ppo_v1/tests/integration/test_v18_encoding_bridge.py \
  riichi_ppo_v1/tests/integration/test_v18_replay_bridge.py \
  riichi_ppo_v1/tests/integration/test_v18_information_boundaries.py \
  riichi_ppo_v1/tests/integration/test_v18_query_semantics.py -q
```

预期：公共/分析/Query 编码与模型 logits、Actor/Critic 隔离断言通过。

## 3. SFT 小规模生命周期

```bash
python -m pytest riichi_ppo_v1/tests/integration/test_v18_sft_lifecycle.py \
  riichi_ppo_v1/tests/unit/test_v18_actor_sft.py -q
```

预期：replay→precompute→shard→collator→Actor-only SFT→保存/加载完成；Critic/value 无梯度；
临时产物清理（pytest tmp_path 自动）。

## 4. 检验/工具

```bash
python -m riichi_ppo_v1.tools.validate --parameter-contract          # 参数与 state keys
python -m riichi_ppo_v1.tools.v18_token_statistics --dataset <临时编码目录> --split validation
python -c "from riichi_ppo_v1.sft.contract import ACTOR_INPUT_CONTRACT_SHA256; print(ACTOR_INPUT_CONTRACT_SHA256)"
```

## 5. 全仓一致性

```bash
rg -n "history_factors|snapshot_factors|_isolated_action_layout|Atomic Snapshot|SNAPSHOT_FIELD_COUNT" \
  riichi_ppo_v1 RiichiEnv/src docs README.md ag -g '*.py' -g '*.md'
```

预期活跃路径零命中；PPO 待迁移引用仅出现在 `training/`、`evaluation/` 与
`audit/reports/v18/report/PROGRESS.md` 的“待迁移”清单。
