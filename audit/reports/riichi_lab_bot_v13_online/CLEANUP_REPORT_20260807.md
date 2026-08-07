# Zenith 清理报告（2026-08-07）

依据 `CLEANUP_TASK_PROMPT.md` 执行，目标仓库为 `/mnt/disk1/hubowen/zenith`。
用户中途明确要求 **保留 ranked 功能**，因此本次清理恢复并保留了 `ranked`
子命令与 `RANKED_URL`；验收过程中没有运行 ranked，也没有连接真实排位端点。

## 一、结论

- `riichi/` 已移入 `RiichiEnv/riichi/`，`import riichi` 公开 API 不变；
  `riichi` 不依赖 `riichienv`。
- `RiichiEnv` 的 ML/UI/WASM/visualizer/agents 已删除，`riichienv` 不再懒加载
  visualizer。
- `riichi_ppo_v1` 的无引用工具、canary 配置、诊断脚本和缓存已删除。
- `riichi_lab_bot` 的 ranked 按用户要求保留；清理期间只执行了 validate。
- 全量回归：**505 passed, 3 skipped**；所有 6.4 专项与线上 validation 通过。
- 清理过程中发现并修复了一个线上 validation bug（服务端
  `riichi_sutehais=None` 与 `riichi_accepted=True` 不一致导致
  `riichi acceptance cannot precede declaration`），修复后
  `validation_passed: true`、0 fallback、0 withheld、0 rejected。

## 二、体积与文件数对比

### 清理前（含构建产物）

| 路径 | 体积 | git 跟踪文件数 |
| --- | ---: | ---: |
| `RiichiEnv/` | 3.8G | 301 |
| `riichi/`（顶层） | 2.1G | 17 |
| `riichi_ppo_v1/` | 3.8M | 103 |
| `riichi_lab_bot/` | 608K | 22 |

### 清理后

| 路径 | 体积 | git 跟踪文件数 |
| --- | ---: | ---: |
| `RiichiEnv/` | 4.7M | 165 |
| `RiichiEnv/riichi/` | 128K | 15 |
| `riichi_ppo_v1/` | 1.1M | 89 |
| `riichi_lab_bot/` | 188K | 22 |

工作树中已不存在 `target/`、`__pycache__/`、`.pytest_cache/`、`*.pyc`、
`*.so`、`*.whl`；`git ls-files` 也不包含这些路径。

## 三、移动与重构

### `riichi/` → `RiichiEnv/riichi/`

- `git mv riichi RiichiEnv/riichi`，保留 crate 名 `riichi`、Python 模块名
  `riichi`、`MjaiKyokuStateMachineManager`、`analyze_features`、
  `ANALYSIS_VERSION == 4`。
- `RiichiEnv/Cargo.toml` workspace 变为：
  `members = ["riichienv-core", "riichienv-python", "riichi"]`。
- 为加入同一 Cargo workspace，`riichi` 的 `pyo3` 从 `0.27.0` 升到 `0.28.0`，
  `numpy` 从 `0.27.1` 升到 `0.28.0`；`cargo check -p riichi` 通过。
- 删除顶层 `riichi` 遗留的嵌套 `.github/workflows/CI.yml` 与独立
  `Cargo.lock`；父 workspace 的 `Cargo.lock` 已重新生成并包含 `riichi`。
- 安装脚本路径已同步：
  `bash RiichiEnv/riichi/scripts/install_conda_extension.sh`。
- 证据：
  - `rg -n "from riichienv|import riichienv" RiichiEnv/riichi` 无结果。
  - `python -c "import riichi; print(riichi.__file__)"` 输出
    `site-packages/riichi/__init__.py`，扩展已从新路径重新安装。
  - `RiichiEnv/tests/test_riichi_package_boundary.py` 3 项通过。

## 四、删除对象与引用证据

### RiichiEnv

先执行全仓库引用扫描，确认引用仅存在于被删目录自身、文档、CI 与测试后，
再 `git rm`：

```text
RiichiEnv/riichienv-ml/      52 files
RiichiEnv/riichienv-ui/      91 files
RiichiEnv/riichienv-wasm/     2 files
RiichiEnv/src/riichienv/visualizer/  3 files
RiichiEnv/src/riichienv/agents/      2 files
RiichiEnv/tests/test_metadata_injection.py
RiichiEnv/docs/SEQUENCE_FEATURE_ENCODING.md
```

同步修改：

- `RiichiEnv/src/riichienv/__init__.py`：删除 `_get_viewer` 懒加载。
- `RiichiEnv/src/riichienv/_riichienv.pyi`：删除 visualizer TYPE_CHECKING。
- `RiichiEnv/pyproject.toml`：删除 visualizer assets include、ruff/ty 的
  ml/ui exclude、`[tool.uv.workspace]`；`uv.lock` 已重新生成（不再含 ml）。
- `RiichiEnv/.github/workflows/ci.yml`：删除 ui-test/ui-lint/wasm-build job
  与 wasm cargo check。
- `RiichiEnv/README.md`、`docs/DEVELOPMENT_GUIDE.md`：重写为核心/riichi
  版本，不再包含 ml/ui/wasm/visualizer/agents。

删除后扫描：

```text
rg -n "riichienv[_-]ml|riichienv[_-]ui|riichienv[_-]wasm|riichienv\.visualizer|riichienv\.agents" RiichiEnv
仅命中 test_riichi_package_boundary.py 的“不应包含”断言。
```

### riichi_ppo_v1

删除前引用扫描显示以下对象只被自身或已删除文档引用：

```text
tools/audit_encoding_pipeline.py
tools/benchmark_ppo_vs_sft.py
tools/benchmark_ppo_vs_sft_detailed.py
tools/benchmark_sft.py
tools/capture_sft_compat_baseline.py
tools/head_to_head_ppo_vs_ppo.py
tools/inspect_mjai_kyoku.py
tools/inspect_sft_npz.py
tools/record_ppo_actions.py
tools/record_sft_vs_heuristic.py
tools/select_sft_and_start_ppo.sh
tools/start_v13_sft_when_ready.sh
tools/train_sft_then_ppo.sh
configs/sft_canary.yaml
tests/diagnose_encoding.py
```

`docs/v13_sft.md` 中的 `tools.benchmark_sft` 命令已替换为
`riichi-ppo-smoke` 说明。

保留：

- `tools/event_statistics.py`：被 `test_event_statistics.py` import。
- `configs/training.yaml`：`load_config()` 默认读取。
- `configs/monitoring.yaml`：`load_config()` 的 `_CONFIG_GROUPS` 读取。
- `configs/sft.yaml`：`sft.train.load_config()` 默认读取。
- `legacy/v11/`：`test_v11_policy_adapter.py` 仍覆盖 V11 兼容。

### riichi_lab_bot

按用户要求 **保留 ranked**，因此 `client.py` 的 `RANKED_URL` / `run_ranked`、
`cli.py` 的 `ranked` 子命令与 `--forever` 均恢复保留。
清理期间没有执行 ranked、没有传 `--forever`、没有连接
`wss://game.riichi.dev/ws/ranked`；只运行了 `validate`。

README 已改为“排位说明”：ranked 供后续线上排位使用，本次清理验收不执行。

## 五、回归与验收

### 5.1 基线（清理前）

```text
502 passed, 3 skipped, 2 warnings in 22.69s
```

### 5.2 清理后全量

```text
505 passed, 3 skipped, 2 warnings in 21.99s
```

测试总数变化说明：

- 删除 `RiichiEnv/tests/test_metadata_injection.py`（visualizer 已删除）。
- 新增 `test_riichi_package_boundary.py` 3 项、`test_cleanup_contract.py`
  3 项、`test_bridge_semantics.py` 1 项。
- 其余删除对象均不是 pytest 测试（`tests/diagnose_encoding.py` 是诊断脚本，
  不参与测试收集）。

### 5.3 6.4 专项

```text
riichi_lab_bot/tests/test_bridge_semantics.py
riichi_lab_bot/tests/test_bridge_integration.py
riichi_ppo_v1/tests/unit/test_semantic_validation.py
riichi_ppo_v1/tests/unit/test_feature_schema_v13.py
riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py
riichi_ppo_v1/tests/protocol/test_protocol_matrix.py
riichi_ppo_v1/tests/integration/test_real_action_cases.py
riichi_ppo_v1/tests/integration/test_bridge_integration.py
=> 46 passed
```

### 5.4 漂移

```text
seed=20260730 decisions=458 elapsed=7.8s
disagreements: 0 / 458 (0.00%)
same actions: 458
```

### 5.5 128 局覆盖

```text
games=128
missing_naturally_observed_events=[]
missing_naturally_observed_action_types=[]
missing_naturally_executed_action_types=[]
```

输出文件：`/mnt/disk1/hubowen/zenith/riichi_ppo_v1_coverage.json`

### 5.6 本地三局 GPU

```text
games=3, warmup_games=1, measured_games=2
measured_decisions=949
measured_decisions_per_second=145.76
fallback_actions=0, withheld_actions=0
```

### 5.7 CPU 冒烟

```text
games=1, decisions=458, fallback_actions=0, withheld_actions=0
```

### 5.8 FP32/BF16

```text
1 passed
```

### 5.9 线上 validation

```text
validation_passed: true
requests=89, responses=89, model_actions=89
fallback_actions=0, withheld_actions=0
accepted=89, rejected=0, unparseable=0, stale=0, defaulted=0
退出码 0
```

日志：`logs/validate_v13_cleanup_fixed_20260807_*.jsonl`

## 六、清理中发现并修复的问题

线上 validation 首次运行出现：

```text
RuntimeError: riichi acceptance cannot precede declaration
```

根因（已通过临时诊断输出确认）：

- 服务端 Observation 提供 `riichi_declared=[True,...]`，但
  `riichi_sutehais=[None,...]`（字段存在但为空）。
- 原桥接逻辑先用原始 `riichi_sutehais=None` 把 `riichi_declared` 修正为
  `False`，再注入事件流重建的 `riichi_accepted=True`，导致语义校验失败。

修复：

- `riichi_lab_bot/src/riichi_lab_bot/bridge.py` 改为以事件流重建的
  `riichi_declared` / `riichi_accepted` / `riichi_declaration_indices` /
  `riichi_sutehais` 为准，统一覆盖服务端不一致的 snapshot 字段。
- 新增回归测试
  `test_server_riichi_snapshot_with_stale_sutehai_and_accepted_missing_prepares`；
  修复前失败，修复后通过。

## 七、保留但“看起来冗余”的文件及理由

| 文件 | 保留理由 |
| --- | --- |
| `riichi_ppo_v1/legacy/v11/` | `test_v11_policy_adapter.py` 覆盖 V11 兼容 |
| `riichi_ppo_v1/tools/event_statistics.py` | `test_event_statistics.py` import |
| `riichi_lab_bot/tools/verify_candidate_token_drift.py` | 6.4 验收工具，漂移 0 |
| `RiichiEnv/riichienv-core/src/observation/sequence_features.rs` | 属于核心 `encode_seq_*` API；注释已去掉 ml 引用 |
| `riichi_lab_bot` ranked 相关代码 | 用户明确要求保留，供后续线上排位 |

## 八、未连接 ranked 声明

本次清理全程没有运行 `riichi-lab-bot ranked`，没有传 `--forever`，没有连接
`wss://game.riichi.dev/ws/ranked`。线上验收只连接
`wss://game.riichi.dev/ws/validate` 并通过。
