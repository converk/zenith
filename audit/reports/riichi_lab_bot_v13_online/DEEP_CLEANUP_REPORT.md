# Zenith 文件内部死代码与遗留逻辑深度清理报告

依据：`audit/reports/riichi_lab_bot_v13_online/DEEP_CLEANUP_TASK_PROMPT.md`
执行时间：2026-08-07
工作区：`/mnt/disk1/hubowen/zenith`

## 1. 结论摘要

对保留文件逐函数/逐类/逐常量/逐参数/逐分支/逐配置项完成内部审计。共检查 333 个保留文件，删除/收敛 30 余处死代码、未使用导入、未使用局部变量、陈旧类型桩、Rust 死代码、重复实现与旧 checkpoint 兼容分支；新增 3 项等价/回归测试；全量测试从 505 增至 506 通过（新增 1 项），覆盖率语句数从 11301 降至 11199。

冻结契约保持：`riichi.ANALYSIS_VERSION == 4`、`riichienv.REPLAY_SEMANTICS_VERSION == 1`；V13 六个状态行、offense/defense 查询对、legal mask、公开 summary 顺序与语义未变；`ranked` 功能完整保留；未删除任何输入资产；未连接 ranked 端点。

## 2. 范围与统计

| 项目 | 数值 |
| --- | ---: |
| 保留文件审计数 | 333 |
| Python 文件 | 183 |
| Rust 文件 | 57 |
| 配置/文档/脚本/锁文件 | 93 |
| 修改文件数 | 35（不含本报告与 `riichi_ppo_v1_coverage.json`） |
| 删除净代码行 | 约 248（`git diff --stat`：-487 / +239，其中含新增测试与文档） |
| 覆盖率语句变化 | 11301 → 11199（-102） |

排除范围：`checkpoints/**`、`datasets/**`、`logs/**` 只读，不清理、不移动。

## 3. 候选死代码总清单与判定

| 候选 | 位置 | 类型 | 判定 |
| --- | --- | --- | --- |
| `sys` | `riichi_lab_bot/src/riichi_lab_bot/cli.py` | 未使用导入 | DELETE |
| `ACTION_QUERY_*` 再导出 | `riichi_ppo_v1/model/schema.py` | 未使用导入/死再导出 | DELETE（调用方改为直接导入 `feature_schema`） |
| `NUM_PLAYERS` 导入、`BLOCK_BY_EVENT`、`MODEL_EVENT_TYPES` | `riichi_ppo_v1/model/validation.py` | 未使用导入/常量 | DELETE |
| `Decision`、`EfficiencyAnalyzer` 导入、`_REPLAY_TYPES` | `riichi_ppo_v1/sft/audit.py` | 未使用导入/常量 | DELETE |
| `encoded_identity_digests` 导入 | `riichi_ppo_v1/tests/unit/test_sft.py` | 未使用导入 | DELETE |
| `numpy` 导入 | `riichi_ppo_v1/tests/unit/test_tensorboard.py` | 未使用导入 | DELETE |
| `policy_loss`、`value_loss_scalar`、`entropy`、`weighted_count` | `riichi_ppo_v1/training/learner.py` | 未使用局部变量 | DELETE |
| `active_envs`、`decisions` | `riichi_lab_bot/tools/verify_candidate_token_drift.py` | 未使用局部变量 | DELETE |
| `scalar_observations` | `riichi_ppo_v1/tests/integration/test_batched_pipeline.py` | 未使用局部变量 | DELETE |
| `metrics` | `riichi_ppo_v1/tests/unit/test_learner.py` | 未使用局部变量 | DELETE |
| `GameViewer` / `get_viewer` 桩 | `RiichiEnv/src/riichienv/_riichienv.pyi` | 陈旧类型桩 | DELETE |
| `_action_key` | `RiichiEnv/scripts/validate_logs.py` | 未使用私有函数 | DELETE |
| `__version__` | `riichi_lab_bot/src/riichi_lab_bot/__init__.py` | 未使用常量 | DELETE |
| `N_TILE_TYPES_4P`、`N_TILES_4P`、`N_TILES_3P` | `RiichiEnv/src/riichienv/consts.py` | 未使用常量 | DELETE（保留被测试使用的 `N_TILE_TYPES_3P`） |
| `SuddenDeathIkkyokuGameMode` | `RiichiEnv/src/riichienv/game_modes.py` | 未使用类 | DELETE |
| `NUM_TILE_TYPES` | `riichi_ppo_v1/model/critic_features.py` | 未使用常量 | DELETE |
| `_public_seat_rows` | `riichi_ppo_v1/model/critic_features.py` | 重复实现 | MERGE（收敛到 `_seat_tiles`） |
| `evaluate_checkpoint_against_heuristics` | `riichi_ppo_v1/sft/heuristic_evaluation.py` | 无引用的公开函数 | DELETE（公开 API，见 4.4） |
| `ActionType.Discard/Chi/...` PascalCase 别名 | `RiichiEnv/src/riichienv/action.py` | 废弃兼容层 | DELETE（公开 API，见 4.4） |
| `tile_id_to_mjai`、`_normalized_action_json`、`snapshot_json`、`action_jsons_and_flag` | `riichi_lab_bot/src/riichi_lab_bot/bridge.py` | 重复实现 | MERGE（收敛到训练端 `model/bridge.py`） |
| `_seat_tiles`、`_meld_field`、`_meld_state`、`_public_meld_rows` | `riichi_lab_bot/src/riichi_lab_bot/features.py` | 重复实现 | MERGE（收敛到 `critic_features.collect_actor_public_table_state`） |
| `greedy`、`record`、`reset_completed` 参数 | `riichi_ppo_v1/training/worker.py::_advance_once` | 唯一调用点固定值参数 | INLINE |
| 旧 v13 checkpoint 格式分支（`contract is None and token_schema_version == 13`） | `riichi_ppo_v1/sft/checkpoint.py`、`sft/policy_adapter.py`、`riichi_lab_bot/.../policy.py` | 旧兼容层 | DELETE（当前 checkpoints 无此格式，见 4.5） |
| `tile_name`、`round_wind_tile`、`jikaze_for` | `RiichiEnv/riichi/src/MjaiKyokuStateMachine/protocol.rs` | Rust 死代码 | DELETE |
| `set_legal_actions` | `RiichiEnv/riichi/src/MjaiKyokuStateMachine/table.rs` | Rust `#[cfg(test)]` 死代码 | DELETE |
| `apply_player_events`、`set_legal_actions` | `RiichiEnv/riichi/src/MjaiKyokuStateMachine/manager.rs` | Rust `#[cfg(test)]` 死代码 | DELETE |
| `checkpoints/checkpoints/train_riichi_v10_sft/best.pt` | `riichi_ppo_v1/README.md` | 死文档路径 | DELETE（改为真实 v13 路径） |

## 4. 删除/收敛证据包

### 4.1 未使用导入与局部变量

引用扫描：`ruff check --select F401,F841,F821` 修改前报告 17 项；修改后 `All checks passed!`。每个符号删除后 `rg` 复扫为 0：

```text
$ rg -n "sys|ACTION_QUERY_DEFENSE|BLOCK_BY_EVENT|_REPLAY_TYPES|GameViewer|_action_key|__version__|N_TILE_TYPES_4P|N_TILES_4P|N_TILES_3P|SuddenDeathIkkyokuGameMode|NUM_TILE_TYPES|evaluate_checkpoint_against_heuristics" ... （省略路径参数）
无匹配
```

### 4.2 Rust 死代码

`cargo check --workspace --all-targets` 修改前输出 5 条 `never used` 告警；删除后：

```text
Checking riichi v0.1.0 (...)
Finished `dev` profile [unoptimized + debuginfo] in 0.69s
```

`manager.rs` 中被删的两个方法位于 `#[cfg(test)] impl`，注释声明“仅供 Rust-only protocol tests”，但 `cargo check --all-targets` 证明无任何测试调用，判定 DELETE。

### 4.3 重复实现收敛

权威实现：`riichi_ppo_v1/model/bridge.py`（token 契约）与 `riichi_ppo_v1/model/critic_features.py`（公开 summary 编码）。

- `riichi_lab_bot/src/riichi_lab_bot/bridge.py`：删除本地 `tile_id_to_mjai`、`_normalized_action_json`、`snapshot_json`、`action_jsons_and_flag`；改为直接导入训练端 `action_jsons_and_decision_flag`、`snapshot_json`、`NUM_PLAYERS`。训练端原私有函数 `_action_jsons_and_decision_flag` 改为公开 `action_jsons_and_decision_flag`，并同步更新 `sft/data.py`、`sft/audit.py`、`legacy/v11/encoder.py`。
- `riichi_lab_bot/src/riichi_lab_bot/features.py`：删除 `_seat_tiles`、`_meld_field`、`_meld_state`、`_public_meld_rows`；新增训练端 `critic_features.collect_actor_public_table_state(observation)` 作为单一适配入口。
- `critic_features.py` 内部 `_seat_tiles` 与 `_public_seat_rows` 行为相同，收敛为一个函数。

等价测试：

```text
$ python -m pytest riichi_ppo_v1/tests/unit/test_bridge_unit.py riichi_lab_bot/tests/test_bridge_integration.py -q
...
19 passed
```

新增 `test_bridge_unit.py` 覆盖 `tile_id_to_mjai` 全部 0–135 tile id；新增 `test_bot_reuses_training_conversion_and_public_summary_helpers` 验证 bot 与训练端 `snapshot_json`、`action_jsons_and_decision_flag`、公开 summary 逐元素相等。

### 4.4 公开 API 删除声明

| 删除的公开符号 | 替代 |
| --- | --- |
| `heuristic_evaluation.evaluate_checkpoint_against_heuristics` | `load_policy_adapter(path)` + `evaluate_against_heuristics(adapter, ...)` |
| `riichienv.action.ActionType.Discard/Chi/...` PascalCase 别名 | `ActionType.DISCARD/CHI/...`（原生枚举成员） |
| `riichienv.consts.N_TILE_TYPES_4P`、`N_TILES_4P`、`N_TILES_3P` | 直接使用字面量 34/136/108；`N_TILE_TYPES_3P` 保留 |
| `riichienv.game_modes.SuddenDeathIkkyokuGameMode` | `OneKyokuGameMode`（等价行为） |

仓库内均无任何代码/测试/配置引用；`riichienv` 为本地训练依赖，未发布给外部使用者。

### 4.5 旧 v13 checkpoint 兼容分支删除

证据：扫描 `checkpoints/` 中全部与 v13 相关 checkpoint，仅 4 个文件，全部带 `sft_contract_version="riichi-sft-v13-1"`：

```text
checkpoints/train_riichi_v13_sft/best.pt          sft_contract_version=riichi-sft-v13-1
checkpoints/train_riichi_v13_sft/best_heuristic.pt sft_contract_version=riichi-sft-v13-1
checkpoints/train_riichi_v13_sft/latest.pt         sft_contract_version=riichi-sft-v13-1
checkpoints/train_riichi_v11_sft_40pct/best_heuristic.pt token_schema_version=11（V11 路径保留）
```

没有任何 checkpoint 使用 `token_schema_version == 13` 且缺少 `sft_contract_version` 的旧格式，因此删除 `load_v13_weights_only`、`load_policy_adapter`、`PolicyEngine` 中的旧格式分支；V11 分支保留。删除后 `rg` 复扫为 0，定向测试 `test_v13_sft_golden.py`、`test_v11_policy_adapter.py`、`test_checkpoint.py` 全部通过。

## 5. 保留的兼容层/旧分支清单及理由

| 保留对象 | 位置 | 理由 |
| --- | --- | --- |
| `_FORMAL_V13_MANIFEST_CONTRACT` 与 `validate_v13_manifest` 旧 tuple 分支 | `riichi_ppo_v1/sft/contract.py` | 当前输入数据集 `datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16/manifest.json` 仍使用该旧四元组格式（`token_schema_version=13`、`feature_schema_sha256=ad8d...`、`rust_analysis_version=4`、`decision_analysis_version=16`）；有测试覆盖 |
| `legacy/v11/` 全部 | `riichi_ppo_v1/legacy/v11/` | V11 checkpoint 存在（`checkpoints/train_riichi_v11_sft_40pct/best_heuristic.pt`），`test_v11_policy_adapter.py` 覆盖 |
| `legacy_fixed` policy head | `riichi_ppo_v1/model/architecture.py` | 冻结 V11 checkpoint 由 legacy adapter 加载所需 |
| `build_legacy_v11` / `_legacy_v11` 参数 | `riichi_ppo_v1/training/rewards/decision.py` | 只由 `legacy/v11` 调用；V11 兼容保留 |
| `RANKED_URL`、`run_ranked`、`ranked` 子命令、`--forever` | `riichi_lab_bot/src/riichi_lab_bot/client.py`、`cli.py` | 用户明确要求保留 ranked |
| ranked 单元测试 | `riichi_lab_bot/tests/test_client.py`、`riichi_ppo_v1/tests/unit/test_cleanup_contract.py` | ranked 保留要求 |
| `mpsz_to_tid_list`、`mpsz_to_mjai_list`、`mjai_to_tid_list`、`mjai_to_mpsz_list` | `RiichiEnv/src/riichienv/convert.py` | 文档化公共转换 API（`docs/DATA_REPRESENTATION.md`），无代码引用但作为对称公共 API 保留 |

## 6. 覆盖率（before/after）

命令：

```bash
python -m coverage run --source=riichi_ppo_v1,riichi_lab_bot,RiichiEnv/src/riichienv,RiichiEnv/riichi \
  -m pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
python -m coverage report --show-missing
```

| 指标 | before | after |
| --- | ---: | ---: |
| 语句总数 | 11301 | 11199 |
| Missing | 2761 | 2715 |
| 覆盖率 | 76% | 76% |

## 7. 静态检查

```text
$ ruff check riichi_ppo_v1 riichi_lab_bot RiichiEnv/src/riichienv --select F401,F841,F821
All checks passed!

$ python -m compileall -q riichi_ppo_v1 riichi_lab_bot RiichiEnv/src/riichienv
COMPILEALL_OK

$ cd RiichiEnv && cargo check --workspace --all-targets
Finished `dev` profile [unoptimized + debuginfo] in 0.69s（无 warning）

$ cd RiichiEnv && LD_LIBRARY_PATH=/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/lib cargo test -p riichi
running 6 tests ... test result: ok. 6 passed; 0 failed
```

完整 `ruff check` 仍报告 165 条纯风格类提示（I001 导入排序、UP 现代语法、B009 等），均不属于死代码；本次不批量修复，避免与“删除死代码”混入无关改动。F 类（未使用/未定义）已清零。

## 8. 等价测试输出

```text
$ python -m pytest riichi_ppo_v1/tests/unit/test_bridge_unit.py \
    riichi_lab_bot/tests/test_bridge_integration.py -q
19 passed

$ CUDA_DEVICE=2,3 python riichi_lab_bot/tools/verify_candidate_token_drift.py \
    --model checkpoints/train_riichi_v13_sft/best_heuristic.pt
seed=20260730 decisions=458 elapsed=7.3s
disagreements: 0 / 458 (0.00%)
same actions: 458
avg bot token len:   97.1
avg train token len: 97.1
```

## 9. 全量回归命令与输出

### 9.1 全量 pytest

```text
$ python -m pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
506 passed, 3 skipped, 2 warnings in 21.60s
```

### 9.2 128 局事件/动作覆盖

```text
$ CUDA_DEVICE=2,3 riichi-ppo-validate --games 128 --seed 20260713 --max-steps 2500 \
    --output riichi_ppo_v1_coverage.json
games = 128
missing_naturally_observed_events = []
missing_naturally_observed_action_types = []
missing_naturally_executed_action_types = []
```

### 9.3 本地三局 GPU（FP32）

```text
$ CUDA_DEVICE=2,3 riichi-lab-bot local --games 3 --seed 20260730 \
    --device cuda:0 --dtype fp32 \
    --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt
{
  "games": 3,
  "warmup_games": 1,
  "measured_games": 2,
  "measured_elapsed_seconds": 6.357469399998081,
  "measured_decisions": 949,
  "measured_decisions_per_second": 149.27323126404454
}
每局 metrics：fallback_actions=0, withheld_actions=0
```

### 9.4 CPU 冒烟

```text
$ riichi-lab-bot local --games 1 --device cpu --dtype fp32 \
    --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt
measured_decisions=458, measured_decisions_per_second=109.57
```

### 9.5 FP32/BF16 一致性

```text
$ CUDA_DEVICE=2,3 python -m pytest riichi_lab_bot/tests/test_checkpoint.py::test_fp32_and_bf16_inference_agree_on_l20 -q
1 passed in 7.29s
```

### 9.6 线上 validation（仅 validate）

```text
$ CUDA_DEVICE=2,3 riichi-lab-bot validate --device cuda:0 --dtype fp32 \
    --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt \
    --jsonl-log logs/validate_v13_deep_cleanup_<ts>.jsonl
{
  "completed": true,
  "validation_passed": true,
  "metrics": {
    "requests": 91, "responses": 91, "model_actions": 91,
    "fallback_actions": 0, "withheld_actions": 0,
    "accepted": 91, "rejected": 0, "unparseable": 0,
    "stale": 0, "defaulted": 0, "bank_consumed_ms": 0
  }
}
```

## 10. ranked 保留声明

`ranked` 功能完整保留：`RANKED_URL`、`run_ranked`、`ranked` 子命令、`--forever`、重连逻辑及其单元测试均未删除。本次清理只运行了 `validate`，未执行 ranked、未传 `--forever`、未连接 `wss://game.riichi.dev/ws/ranked`；新增代码与日志中均无 ranked 调用。

## 11. 挂起/不确定候选

无 UNCERTAIN 删除。以下对象按“保留并说明理由”处理，不视为挂起：

- `riichienv.convert` 四个列表转换函数：无代码引用但有文档化公共 API，保留。
- `riichienv.convert.tid_to_mjai` 与 `riichi_ppo_v1.model.bridge.tile_id_to_mjai` 行为等价，但分属环境公共转换层与模型 token 边界层，依赖方向相反，按不同契约隔离，不强行合并。
- `riichi_ppo_v1/sft/contract.py` 旧 manifest tuple：当前数据集仍在用，保留。
- 完整 `ruff` 的 165 条风格提示：非死代码，未处理，供后续独立 lint 任务处理。

## 12. 全仓库逐文件内部审计表

下表覆盖全部 333 个保留文件。判定列取值：`KEEP`（保留，未发现内部死代码）、`已清理`（本任务删除/收敛内部符号）、`KEEP_COMPAT`（保留兼容对象）。职责摘要取自模块 docstring 首行；无 docstring 的配置/文档按类别说明。

| 文件 | 类别 | 职责摘要 | 判定 | 说明 |
| --- | --- | --- | --- | --- |
| `.gitignore` | 其他 |  | KEEP |  |
| `.vscode/settings.json` | JSON 数据/输出 |  | KEEP |  |
| `AGENTS.md` | 文档 |  | KEEP |  |
| `RiichiEnv/.cargo/config.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/.github/workflows/ci.yml` | 配置 |  | KEEP |  |
| `RiichiEnv/.github/workflows/release.yml` | 配置 |  | KEEP |  |
| `RiichiEnv/.gitignore` | 其他 |  | KEEP |  |
| `RiichiEnv/.pre-commit-config.yaml` | 配置 |  | KEEP |  |
| `RiichiEnv/.vscode/extensions.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/.vscode/settings.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/CONTRIBUTING.md` | 文档 |  | KEEP |  |
| `RiichiEnv/Cargo.lock` | 锁文件 |  | KEEP |  |
| `RiichiEnv/Cargo.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/LICENSE` | 其他 |  | KEEP |  |
| `RiichiEnv/README.md` | 文档 |  | KEEP |  |
| `RiichiEnv/commitlint.config.js` | 其他 |  | KEEP |  |
| `RiichiEnv/docs/DATA_REPRESENTATION.md` | 文档 |  | KEEP |  |
| `RiichiEnv/docs/DEVELOPMENT_GUIDE.md` | 文档 |  | KEEP |  |
| `RiichiEnv/docs/ENCODING.md` | 文档 |  | KEEP |  |
| `RiichiEnv/docs/FEATURE_ENCODING.md` | 文档 |  | KEEP |  |
| `RiichiEnv/docs/RULES.md` | 文档 |  | KEEP |  |
| `RiichiEnv/docs/assets/logo.jpg` | 其他 |  | KEEP |  |
| `RiichiEnv/docs/assets/visualizer1.png` | 其他 |  | KEEP |  |
| `RiichiEnv/docs/assets/visualizer2.png` | 其他 |  | KEEP |  |
| `RiichiEnv/pyproject.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/renovate.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/riichi/.gitignore` | 其他 |  | KEEP |  |
| `RiichiEnv/riichi/Cargo.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/riichi/pyproject.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/riichi/scripts/install_conda_extension.sh` | Shell 脚本 |  | KEEP |  |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/manager.rs` | Rust 源码 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/mod.rs` | Rust 源码 | Append-only MJAI-to-model state machines for one four-player kyoku per table. | KEEP |  |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/player.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/protocol.rs` | Rust 源码 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/semantic_token_tests.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/table.rs` | Rust 源码 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/riichi/src/MjaiKyokuStateMachine/types.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichi/src/analysis.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichi/src/lib.rs` | Rust 源码 | Python entry point for the MJAI kyoku state-machine extension. | KEEP |  |
| `RiichiEnv/riichi/src/shanten.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichi/src/shanten_table.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/Cargo.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/riichienv-core/README.md` | 文档 |  | KEEP |  |
| `RiichiEnv/riichienv-core/benches/agari_bench.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/benches/data/agari_3p.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/riichienv-core/benches/data/agari_4p.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/riichienv-core/benches/data/hands_negative.json` | JSON 数据/输出 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/action.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/agari.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/data/nyanten_keys1.bin` | 其他 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/data/nyanten_keys2.bin` | 其他 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/data/nyanten_keys3.bin` | 其他 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/data/nyanten_shupai_keys.bin` | 其他 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/data/nyanten_zipai_keys.bin` | 其他 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/errors.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/game_variant.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/hand_evaluator.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/hand_evaluator_3p.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/lib.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/encode.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/helpers.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/mjai_select.rs` | Rust 源码 | Shared helpers for mapping Mjai messages to a legal `Action`. | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/mod.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/python.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation/sequence_features.rs` | Rust 源码 | Sequence feature encoding for transformer models. | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation_3p/encode.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation_3p/helpers.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation_3p/mod.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/observation_3p/python.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/parser.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/replay/mjai_replay.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/replay/mjsoul_replay.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/replay/mod.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/rule.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/score.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/shanten.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/event_handler.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/game_mode.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/legal_actions.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/mod.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/player.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state/wall.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/event_handler.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/game_mode.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/legal_actions.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/mod.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/player.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/sanma.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/state_3p/wall.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/tests.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/types.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/yaku.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/yaku_3p.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/src/yaku_checker.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-core/tests/agari_correctness.rs` | Rust 测试 | Correctness tests for agari benchmark data. | KEEP |  |
| `RiichiEnv/riichienv-python/Cargo.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/riichienv-python/src/env.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/riichienv-python/src/lib.rs` | Rust 源码 |  | KEEP |  |
| `RiichiEnv/rust-toolchain.toml` | 包配置 |  | KEEP |  |
| `RiichiEnv/scripts/install_conda_extension.sh` | Shell 脚本 |  | KEEP |  |
| `RiichiEnv/scripts/repro_encode_shanten_efficiency_houou.py` | Python 脚本/工具 | Reproduce the Houou shanten-efficiency panic from a Tenhou JSON paipu. | KEEP |  |
| `RiichiEnv/scripts/validate_logs.py` | Python 脚本/工具 | Datasource: https://www.kaggle.com/datasets/shokanekolouis/tenhou-to-mjai | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/src/riichienv/__init__.py` | Python 源码 |  | KEEP |  |
| `RiichiEnv/src/riichienv/_riichienv.pyi` | Python 类型桩 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/src/riichienv/action.py` | Python 源码 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/src/riichienv/consts.py` | Python 源码 | Tile-dimension and encoding constants for RiichiEnv. | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/src/riichienv/convert.py` | Python 源码 |  | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `RiichiEnv/src/riichienv/game_mode.py` | Python 源码 |  | KEEP |  |
| `RiichiEnv/src/riichienv/game_modes.py` | Python 源码 |  | 已清理 | 本任务修改；见证据包 |
| `RiichiEnv/src/riichienv/hand.py` | Python 源码 |  | KEEP |  |
| `RiichiEnv/tests/__init__.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/data/126_204_0_mjai.jsonl` | 其他 |  | KEEP |  |
| `RiichiEnv/tests/env/__init__.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/__init__.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_action_to_mjai.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_daiminkan_rinshan_draw.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_kakan.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_kyushu_kyuhai.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_meld_aka.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_relaxed_red5.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_riichi_autoplay_pass.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_riichi_no_claim.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/actions/test_riichi_pass.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/agari/__init__.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/agari/test_chankan.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/agari/test_pao.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/agari/test_pao_honba.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/helper.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/__init__.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_claim_priority.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_furiten_rules.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_kuikae.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_riichi_sequence.py` | Python 测试 | Test for Riichi action sequence handling. | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_temporary_furiten.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/rule_validation/test_valid_ankan.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_apply_event.py` | Python 测试 | Integration tests for RiichiEnv.observe_event() API. | KEEP |  |
| `RiichiEnv/tests/env/test_apply_event_mjai_log.py` | Python 测试 | Tests for mjai_log recording in apply_event() and observe_event(). | KEEP |  |
| `RiichiEnv/tests/env/test_discard_type.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_env_ranks_points.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_game_modes.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_game_rules.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_honba_reset.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_illegal_actions.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_kan_dora_timing_events.py` | Python 测试 | Test kan dora reveal timing event ordering | KEEP |  |
| `RiichiEnv/tests/env/test_m263_ron_mismatch.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_majsoul_pao_scoring.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_paishan.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_reset_game.py` | Python 测试 | Regression tests: env.reset() must start a fresh game. | KEEP |  |
| `RiichiEnv/tests/env/test_riichi_markers.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_riichi_no_claims.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_riichienv.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_riichienv_hora.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_rules_chankan.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/env/test_sanma.py` | Python 测试 | Tests for 3-player (sanma) mahjong via RiichiEnv. | KEEP |  |
| `RiichiEnv/tests/test_agari_calculator.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_calculate_score.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_convert.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_core.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_env_scoring.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_midway_draw.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_mjai_parity.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_mjai_replay.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_observation_serialization.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_oyayame_tiebreak.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_riichi_autoplay.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_riichi_package_boundary.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/tests/test_shanten.py` | Python 测试 |  | KEEP |  |
| `RiichiEnv/uv.lock` | 锁文件 |  | KEEP |  |
| `audit/reports/riichi_lab_bot_v13_online/CLEANUP_REPORT_20260807.md` | 文档 |  | KEEP | 审计任务/报告文件 |
| `audit/reports/riichi_lab_bot_v13_online/CLEANUP_TASK_PROMPT.md` | 文档 |  | KEEP | 审计任务/报告文件 |
| `audit/reports/riichi_lab_bot_v13_online/DEEP_CLEANUP_REPORT.md` | 文档 |  | KEEP |  |
| `audit/reports/riichi_lab_bot_v13_online/DEEP_CLEANUP_TASK_PROMPT.md` | 文档 |  | KEEP | 审计任务/报告文件 |
| `audit/reports/riichi_lab_bot_v13_online/TASK_PROMPT.md` | 文档 |  | KEEP | 审计任务/报告文件 |
| `audit/reports/riichi_lab_bot_v13_online/riichi_ppo_v1_coverage.json` | JSON 数据/输出 |  | KEEP | 审计任务/报告文件 |
| `audit/reports/sft_contract_cleanup_20260802/REPORT.md` | 文档 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/REPORT.md` | 文档 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/action_coverage.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/action_roundtrip_random.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/actor_only_runtime.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/benchmark.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/commands.txt` | 说明/依赖 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/ddp_loss_weight.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/ddp_plan.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/environment.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/git_status.txt` | 说明/依赖 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/heuristic_comparison.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/heuristic_run_1.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/heuristic_run_2.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/houtei_search.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/identity_digest.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/manifest_scan.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/model_checkpoint.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/rare_event_search.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/rare_replay_comparison.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/raw_selection.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/replay_comparison.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/replay_full_kyoku_comparison.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/replay_sample_identities.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/resume_comparison.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/action_roundtrip_random.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/actor_only_runtime.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/benchmark_sft_timed.py` | Python 脚本/工具 | Benchmark a bounded number of real SFT optimizer steps. | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/ddp_loss_weight_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/ddp_plan_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/full_scan_parallel.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/heuristic_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/houtei_search.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/model_checkpoint_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/rare_event_search.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/rare_replay_compare.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/raw_selection_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/replay_compare.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/replay_full_kyoku_compare.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/resume_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/resume_audit_fixed_horizon.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/special_event_coverage.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/scripts/validation_audit.py` | Python 脚本/工具 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/special_event_coverage.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/test_results.json` | JSON 数据/输出 |  | KEEP |  |
| `audit/reports/v13_sft_20260802/validation.json` | JSON 数据/输出 |  | KEEP |  |
| `requirements.txt` | 说明/依赖 |  | KEEP |  |
| `riichi_lab_bot/.gitignore` | 其他 |  | KEEP |  |
| `riichi_lab_bot/README.md` | 文档 |  | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_lab_bot/pyproject.toml` | 包配置 |  | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/__init__.py` | Python 源码 | Standalone RiichiLab bot for Zenith checkpoints. | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/src/riichi_lab_bot/bootstrap.py` | Python 源码 | Import-light console bootstrap that installs CUDA visibility first. | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/bridge.py` | Python 源码 | Single-seat RiichiEnv observation to semantic-token bridge. | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/src/riichi_lab_bot/cli.py` | Python 源码 | Command-line entry points. | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/src/riichi_lab_bot/client.py` | Python 源码 | RiichiLab WebSocket validation and ranked clients. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_lab_bot/src/riichi_lab_bot/features.py` | Python 源码 | Actor-visible public river and meld summary. | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/src/riichi_lab_bot/local_play.py` | Python 源码 | Local RiichiEnv simulation through the online-shaped bridge. | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/model.py` | Python 源码 | Canonical V13 model exports shared with the training code path. | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/observation.py` | Python 源码 | Online RiichiEnv observation normalization for the lab server schema. | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/policy.py` | Python 源码 | Checkpoint loading and deterministic policy inference. | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/src/riichi_lab_bot/safety.py` | Python 源码 | MJAI response construction and chombo-avoidance checks. | KEEP |  |
| `riichi_lab_bot/src/riichi_lab_bot/telemetry.py` | Python 源码 | Structured, secret-safe runtime metrics. | KEEP |  |
| `riichi_lab_bot/tests/conftest.py` | Python 测试 |  | KEEP |  |
| `riichi_lab_bot/tests/test_bridge_integration.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/tests/test_bridge_semantics.py` | Python 测试 | Bot bridge semantic edge matrix for V13 online observations. | KEEP |  |
| `riichi_lab_bot/tests/test_checkpoint.py` | Python 测试 |  | KEEP |  |
| `riichi_lab_bot/tests/test_client.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_lab_bot/tests/test_safety.py` | Python 测试 |  | KEEP |  |
| `riichi_lab_bot/tools/verify_candidate_token_drift.py` | Python 脚本/工具 | Empirically verify the bot and training paths emit identical V13 tokens. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/README.md` | 文档 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/__init__.py` | Python 源码 | RiichiEnv PPO semantic-token Transformer training package. | KEEP |  |
| `riichi_ppo_v1/configs/monitoring.yaml` | 配置 |  | KEEP |  |
| `riichi_ppo_v1/configs/sft.yaml` | 配置 |  | KEEP |  |
| `riichi_ppo_v1/configs/training.yaml` | 配置 |  | KEEP |  |
| `riichi_ppo_v1/docs/KyokuActionSpace.md` | 文档 |  | KEEP |  |
| `riichi_ppo_v1/docs/KyokuEventTupleProtocol.md` | 文档 |  | KEEP |  |
| `riichi_ppo_v1/docs/v13_sft.md` | 文档 |  | KEEP |  |
| `riichi_ppo_v1/legacy/__init__.py` | Python 兼容层 | Frozen compatibility islands excluded from current training paths. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/legacy/v11/__init__.py` | Python 兼容层 | Frozen schema-11 inference support. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/legacy/v11/adapter.py` | Python 兼容层 | V11 implementation of the shared evaluation policy interface. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/legacy/v11/contract.py` | Python 兼容层 | Fixed identifiers for the frozen v11 inference contract. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/legacy/v11/encoder.py` | Python 兼容层 | Frozen online feature encoder used only for v11 checkpoint evaluation. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/legacy/v11/model.py` | Python 兼容层 | Strict weights-only loader for historical v11 checkpoints. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/model/__init__.py` | Python 源码 | Model architecture and model/environment conversion boundary. | KEEP |  |
| `riichi_ppo_v1/model/actor_features.py` | Python 源码 | Compact deterministic public-state summaries for v13 actor inputs. | KEEP |  |
| `riichi_ppo_v1/model/architecture.py` | Python 源码 | Semantic-token decoder-only GQA actor-critic. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/model/bridge.py` | Python 源码 | Strict conversion boundary between RiichiEnv, MJAI and the Rust state machine. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/model/critic_features.py` | Python 源码 | Centralized-value and shared public-summary features. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/model/dora.py` | Python 源码 | Shared public dora-indicator semantics for actor feature encoding. | KEEP |  |
| `riichi_ppo_v1/model/feature_schema.py` | Python 源码 | Frozen semantic field contract for actor feature schema 13. | KEEP |  |
| `riichi_ppo_v1/model/schema.py` | Python 源码 | Versioned serialization constants shared by PPO and SFT. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/model/semantic_validation.py` | Python 源码 | Semantic assertions and readable summaries for model input tokens. | KEEP |  |
| `riichi_ppo_v1/model/validation.py` | Python 源码 | Reusable validation helpers for the 4-player semantic-token integration boundary. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/pyproject.toml` | 包配置 |  | KEEP |  |
| `riichi_ppo_v1/sft/__init__.py` | Python 源码 | Offline MJAI preparation and supervised actor-critic training. | KEEP |  |
| `riichi_ppo_v1/sft/audit.py` | Python 源码 | Deterministic end-to-end integrity checks for compact SFT preprocessing. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/sft/checkpoint.py` | Python 源码 | Explicit v13 SFT exact-resume and weights-only checkpoint loaders. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/sft/contract.py` | Python 源码 | Single fail-closed contract boundary for the supported v13 SFT path. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/sft/data.py` | Python 源码 | Online kyoku replay and model-input encoding for SFT. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/sft/evaluation_cases.py` | Python 源码 | Deterministic, seat-balanced schedules for heuristic policy evaluation. | KEEP |  |
| `riichi_ppo_v1/sft/head_to_head.py` | Python 源码 | Deterministic, seat-balanced 2v2 checkpoint evaluation. | KEEP |  |
| `riichi_ppo_v1/sft/heuristic_evaluation.py` | Python 源码 | In-process SFT policy evaluation against fixed heuristic opponents. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/sft/policy_adapter.py` | Python 源码 | Version-independent policy boundary for deterministic evaluation. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/sft/precompute.py` | Python 源码 | Materialize a deterministic compact, actor-only SFT subset for fast training. | KEEP |  |
| `riichi_ppo_v1/sft/prepare.py` | Python 源码 | Prepare yearly tenhou-to-mjai archives as replayable kyoku tar shards. | KEEP |  |
| `riichi_ppo_v1/sft/tensorboard.py` | Python 源码 | Low-overhead TensorBoard metrics for supervised riichi training. | KEEP |  |
| `riichi_ppo_v1/sft/train.py` | Python 源码 | Joint supervised policy/value training over prepared MJAI kyoku shards. | KEEP |  |
| `riichi_ppo_v1/tests/__init__.py` | Python 测试 | Riichi PPO regression tests. | KEEP |  |
| `riichi_ppo_v1/tests/integration/__init__.py` | Python 测试 | Tests that require the RiichiEnv and Rust extensions. | KEEP |  |
| `riichi_ppo_v1/tests/integration/test_batched_pipeline.py` | Python 测试 | Batch-native state and environment contracts used by the PPO worker. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/tests/integration/test_bridge_integration.py` | Python 测试 | Requires locally built ``riichi`` and ``riichienv`` extensions. | KEEP |  |
| `riichi_ppo_v1/tests/integration/test_real_action_cases.py` | Python 测试 | Controlled RiichiEnv legal windows must map and decode through the 241-space. | KEEP |  |
| `riichi_ppo_v1/tests/integration/test_v11_policy_adapter.py` | Python 测试 |  | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/tests/integration/test_v13_sft_golden.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/protocol/__init__.py` | Python 测试 | Action-space and semantic-token protocol regression tests. | KEEP |  |
| `riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py` | Python 测试 | Exhaustively validate all 241 fixed action slots through the public binding. | KEEP |  |
| `riichi_ppo_v1/tests/protocol/test_protocol_matrix.py` | Python 测试 | Semantic-token matrix tests for the public Python state-machine API. | KEEP |  |
| `riichi_ppo_v1/tests/unit/__init__.py` | Python 测试 | Fast isolated tests for the training package. | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_bridge_unit.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/tests/unit/test_cleanup_contract.py` | Python 测试 |  | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/tests/unit/test_config_loading.py` | Python 测试 | Configuration-layer loading and precedence tests. | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_critic_features.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_decision_analysis.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_event_statistics.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_feature_schema_v13.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_head_to_head.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_heuristic_policy.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_inference_batching.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_learner.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/tests/unit/test_metrics.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_model.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_ppo_evaluation.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_public_state.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_semantic_validation.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_sft.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/tests/unit/test_sft_contract.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_sft_tensorboard.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/unit/test_tensorboard.py` | Python 测试 |  | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/tests/unit/test_trajectory.py` | Python 测试 |  | KEEP |  |
| `riichi_ppo_v1/tests/validate.py` | Python 测试 | Run strict real-environment validation for semantic state and action protocols. | KEEP |  |
| `riichi_ppo_v1/tools/event_statistics.py` | Python 脚本/工具 | Step-scoped event identity and detailed 2v2 metric primitives. | KEEP |  |
| `riichi_ppo_v1/training/__init__.py` | Python 源码 | PPO rollout, inference, optimisation and command-line entry points. | KEEP |  |
| `riichi_ppo_v1/training/evaluation.py` | Python 源码 | Configuration and metric adapters for periodic PPO evaluation. | KEEP |  |
| `riichi_ppo_v1/training/inference.py` | Python 源码 | GPU PPO actors with cross-worker rollout inference batches. | KEEP |  |
| `riichi_ppo_v1/training/learner.py` | Python 源码 | PPO optimisation and variable-length batch collation. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1/training/metrics.py` | Python 源码 | Semantic, privacy-safe training metrics for Riichi PPO. | KEEP |  |
| `riichi_ppo_v1/training/opponents/__init__.py` | Python 源码 | Evaluation-only public-information heuristic opponents. | KEEP |  |
| `riichi_ppo_v1/training/opponents/heuristic.py` | Python 源码 | CPU-only, public-information heuristic opponents. | KEEP |  |
| `riichi_ppo_v1/training/profiling.py` | Python 源码 | Low-overhead stage timing and GPU telemetry for PPO bottleneck analysis. | KEEP |  |
| `riichi_ppo_v1/training/rewards/__init__.py` | Python 源码 | Local rollout reward components; none of these alter environment rules. | KEEP |  |
| `riichi_ppo_v1/training/rewards/decision.py` | Python 源码 | Rule-aware public decision analysis shared by rollout, teachers and opponents. | KEEP_COMPAT | 保留兼容对象；见保留清单 |
| `riichi_ppo_v1/training/rewards/efficiency.py` | Python 源码 | Cached, batched shanten-first public-ukeire discard rewards. | KEEP |  |
| `riichi_ppo_v1/training/rewards/public_state.py` | Python 源码 | Incremental public-tile accounting for rewards and heuristic defence. | KEEP |  |
| `riichi_ppo_v1/training/rewards/terminal.py` | Python 源码 | Terminal kyoku reward scaling shared by rollout workers and tests. | KEEP |  |
| `riichi_ppo_v1/training/tensorboard.py` | Python 源码 | Curated real-time TensorBoard projection for Riichi PPO. | KEEP |  |
| `riichi_ppo_v1/training/train.py` | Python 源码 | Command-line entry points for synchronous Ray PPO training. | KEEP |  |
| `riichi_ppo_v1/training/trajectory.py` | Python 源码 | Rollout records and kyoku-local GAE. | KEEP |  |
| `riichi_ppo_v1/training/worker.py` | Python 源码 | Ray rollout actors.  Workers own environments and Rust state-machine slots. | 已清理 | 本任务修改；见证据包 |
| `riichi_ppo_v1_coverage.json` | JSON 数据/输出 |  | KEEP | 本任务重新生成（128 局覆盖输出） |
