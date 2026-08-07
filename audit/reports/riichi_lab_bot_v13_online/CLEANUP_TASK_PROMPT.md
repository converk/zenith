# 任务：Zenith 全仓库整理与清洁（详细执行版）

## 0. 文档用途

本文档是“全仓库冗余清理 + 包边界重构 + 严格回归”的执行提示词，供执行代理逐条照做。它不依赖执行代理对项目的既有印象；所有结论必须来自当前工作树、git 历史、测试输出和引用扫描。

配套文档：

- `DEEP_CLEANUP_TASK_PROMPT.md`：文件内部的逐函数/逐分支/逐配置项深度审计；
- `TASK_PROMPT.md`：上一轮 V13 bot 上线任务的原始要求与验证阶梯；
- `AGENTS.md`：项目环境、GPU、目录约定。

## 1. 任务目标

1. 对整个仓库进行逐文件审计，覆盖源码、测试、配置、文档、脚本、CI、构建产物、缓存、示例、历史遗留代码。
2. 删除确认无用且无引用的文件、目录、构建产物、缓存和历史残留。
3. 将顶层独立 crate `riichi/` 移入 `RiichiEnv/riichi/`，作为 RiichiEnv 的子包/子 crate，负责“MJAI 协议状态转换与保存”，与环境本身解耦，同时保持 `import riichi` 的公开 API 不变。
4. 消除全仓库范围内重复、漂移或职责混乱的实现，收敛到唯一权威实现。
5. 在不改变 V13 输入契约、不破坏运行时契约、不删除输入资产、不破坏 ranked 功能的前提下完成清理。

## 2. 非目标

- 不重训模型、不转换 checkpoint、不修改 V13 输入契约；
- 不新增训练算法、不新增 bot 功能；
- 不重写 RiichiLab 协议；
- 不删除 `ranked` 功能；
- 不清理 `checkpoints/`、`datasets/`；
- 不做与清理无关的大规模重命名或架构重写。

## 3. 范围

### 3.1 纳入范围

仓库根目录下所有内容，包括但不限于：

```text
RiichiEnv/
riichi/
riichi_ppo_v1/
riichi_lab_bot/
exp/
audit/
docs/
scripts/
configs/
.github/
AGENTS.md
README.md
requirements.txt
所有 pyproject.toml / Cargo.toml / Cargo.lock / uv.lock / package-lock.json
所有 .yaml / .toml / .json / .sh / .rs / .py / .md / .ipynb 文件
所有 target / __pycache__ / .pytest_cache / build / dist / wheels
```

### 3.2 只读范围

```text
checkpoints/**
datasets/**
logs/**
```

这些目录只用于验证命令读取，不清理、不移动、不修改。

### 3.3 保留范围

```text
ranked 功能相关代码：
- riichi_lab_bot/src/riichi_lab_bot/client.py 中的 RANKED_URL、run_ranked
- riichi_lab_bot/src/riichi_lab_bot/cli.py 中的 ranked 子命令、--forever
- 相关单元测试
```

保留不代表本任务运行它；本任务期间禁止运行真实 ranked 命令和连接真实 ranked 端点。

## 4. 硬性约束

| 编号 | 约束 |
| --- | --- |
| C1 | `riichi.ANALYSIS_VERSION == 4`、`riichienv.REPLAY_SEMANTICS_VERSION == 1` 必须保持，任何修改不得破坏。 |
| C2 | V13 输入契约不可变：六个 state 行、每个合法动作的 offense/defense 查询对、legal mask、public summary 的顺序与语义必须与 `riichi-sft-v13-1` 一致。 |
| C3 | `checkpoints/`、`datasets/` 只读，禁止删除、移动、改名、重训、转换。 |
| C4 | `ranked` 功能必须保留：代码、URL、CLI、重连逻辑、单元测试均不得删除；本任务不得运行真实 ranked。 |
| C5 | 每个删除动作必须有证据：先引用扫描，再删除，删除后全量回归。 |
| C6 | 小步提交：每个提交只包含一个主题（删除一个文件、移动一个 crate、收敛一个重复函数）。 |
| C7 | 不确定的候选必须“挂起”并报告，不得删除。 |
| C8 | 不允许只清理提示词中举例的文件；示例只是起点，范围以整个仓库为准。 |
| C9 | 公共 API 的删除必须显式声明；涉及外部使用者时，保留兼容层或先征求用户同意。 |
| C10 | 不得为了通过测试而修改测试期望，除非该测试断言的是已废弃行为且证据充分。 |

## 5. 执行前审计

### 5.1 全仓库文件清单

```bash
cd /mnt/disk1/hubowen/zenith

rg --files \
  -g '!checkpoints/**' -g '!datasets/**' -g '!logs/**' \
  -g '!**/target/**' -g '!**/__pycache__/**' -g '!**/.pytest_cache/**' \
  | sort > /tmp/repo_files.txt

wc -l /tmp/repo_files.txt
```

### 5.2 按类型统计

```bash
for ext in py rs toml yaml json md sh ipynb lock; do
  count=$(rg --files -g "*.$ext" | wc -l)
  echo "$ext $count"
done
```

### 5.3 体积与构建产物

```bash
du -sh -- * 2>/dev/null | sort -h
du -sh RiichiEnv/target riichi/target 2>/dev/null || true

find . -path '*/target' -prune -o -name '__pycache__' -print | head -200
find . -name '.pytest_cache' -type d -print | head -100
find . -name '*.pyc' -print | head -200
find . -name '*.so' -print | head -100
find . -name '*.whl' -print | head -100
git ls-files | rg '(target/|__pycache__/|\.pytest_cache/|\.so$|\.whl$)' || true
```

### 5.4 输出《全仓库冗余清单》

清单必须包含：

| 路径 | 类型 | 大小 | 最后修改 | 引用数 | 测试覆盖 | 当前运行依赖 | 判定 | 风险 |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |

判定取值：

```text
KEEP                  保留
KEEP_REFACTOR         保留但需要内部清理（转 DEEP_CLEANUP）
MOVE                  移动到新位置
DELETE                删除
DUPLICATE             与另一文件重复，收敛后删除
LEGACY                历史兼容，需决定保留或删除
UNCERTAIN             无法确定，挂起并报告
ARTIFACT              构建/缓存产物，删除
INPUT_ASSET           输入资产，禁止修改
```

### 5.5 引用扫描方法

对每个候选对象执行以下四类扫描：

```bash
# 1) 源码引用
rg -n "候选对象名/路径" \
  -g '!**/target/**' -g '!**/__pycache__/**' \
  -g '!checkpoints/**' -g '!datasets/**' -g '!logs/**' .

# 2) 动态/字符串引用
rg -n '"候选对象名"|'"'"'候选对象名'"'"'' \
  --glob '*.py' --glob '*.yaml' --glob '*.toml' --glob '*.json' \
  --glob '*.rs' --glob '*.sh' --glob '*.md' .

# 3) git 追踪状态
git ls-files --error-unmatch "候选对象" 2>&1 || true
git log --oneline -5 -- "候选对象" 2>&1 || true

# 4) 安装/入口/CI
rg -n "候选对象名" pyproject.toml Cargo.toml .github AGENTS.md README.md \
  riichi*/**/*.toml RiichiEnv/**/*.toml 2>/dev/null || true
```

## 6. 分类决策树

对每个文件/目录/符号按以下顺序判断：

```text
1. 是否属于 checkpoints/datasets/logs？
   → 是：不处理。

2. 是否属于 ranked 功能？
   → 是：保留；只允许内部清理，不允许删除功能。

3. 是否是构建/缓存产物？
   → 是：从工作树和 git 索引删除，提交为 ARTIFACT。

4. 是否被当前代码、测试、配置、文档、CI、脚本、动态调用引用？
   → 否：标记 DELETE，进入删除流程。
   → 是：继续。

5. 是否属于冻结契约（V13 schema、241 动作空间、ANALYSIS_VERSION、REPLAY_SEMANTICS_VERSION）？
   → 是：保留，禁止改动语义。

6. 是否与另一处实现重复？
   → 是：标记 DUPLICATE，选择权威实现，收敛后删除副本。

7. 是否属于旧版本兼容（V11、旧 checkpoint、旧 manifest、旧配置）？
   → 是：检查当前输入资产是否仍需要；需要则保留并说明，不需要则进入删除流程。

8. 是否只被测试引用，且测试本身只验证旧行为？
   → 是：候选删除测试与实现，进入删除流程。

9. 是否无法判断？
   → 是：标记 UNCERTAIN，挂起并报告，不删除。
```

## 7. 删除流程（必须按顺序执行）

### 7.1 文件/目录删除

```bash
# Step 1: 引用扫描必须为 0
rg -n "候选对象" -g '!**/target/**' -g '!**/__pycache__/**' \
  -g '!checkpoints/**' -g '!datasets/**' -g '!logs/**' . || true

# Step 2: 记录删除前状态
git status --short

# Step 3: 删除
git rm -r "候选对象"   # 或 rm -rf 后 git add -A

# Step 4: 删除后复扫
rg -n "候选对象" -g '!**/target/**' -g '!**/__pycache__/**' \
  -g '!checkpoints/**' -g '!datasets/**' -g '!logs/**' . || true

# Step 5: 运行受影响模块的定向测试
pytest <受影响测试> -q

# Step 6: 运行全量测试
pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
```

### 7.2 符号删除

符号删除必须额外满足：

- 函数/类/常量名在源码、测试、文档、CI、动态引用中均为 0；
- `getattr`、`globals()`、`__getattribute__`、字符串 key、YAML key、CLI 参数均无引用；
- 删除后 `python -c "import <模块>"` 仍成功；
- 删除后全量测试通过；
- 如果是公开 API，必须在报告中声明“外部兼容影响”。

### 7.3 不确定候选

```text
如果引用扫描不为 0，但引用点本身也疑似死代码：
→ 先审计引用点，再决定；不要同时删除两个互相依赖的可疑对象。

如果无法判断是否属于冻结契约：
→ 标记 UNCERTAIN，写入报告，等待用户确认。

如果删除会影响 ranked 功能：
→ 一律保留。
```

## 8. 移动/重构流程（以 `riichi` 移入 RiichiEnv 为例）

### 8.1 移动前

```bash
# 记录移动前 API 与测试基线
python -c "import riichi; print(riichi.__file__)"
python -c "import riichi; print(riichi.ANALYSIS_VERSION)"
pytest riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py \
  riichi_ppo_v1/tests/protocol/test_protocol_matrix.py \
  riichi_ppo_v1/tests/unit/test_feature_schema_v13.py -q
```

### 8.2 移动

```bash
mkdir -p RiichiEnv/riichi
git mv riichi/* RiichiEnv/riichi/
git mv riichi/.gitignore RiichiEnv/riichi/.gitignore
git mv riichi/.github RiichiEnv/riichi/.github 2>/dev/null || true
```

### 8.3 同步更新

必须检查并更新：

- `RiichiEnv/Cargo.toml` workspace members；
- `riichi/Cargo.toml` → `RiichiEnv/riichi/Cargo.toml`；
- `riichi/pyproject.toml` → `RiichiEnv/riichi/pyproject.toml`；
- 所有安装脚本：`riichi/scripts/install_conda_extension.sh` 等；
- `AGENTS.md`、`README.md`、CI 中所有 `cd riichi` / `riichi/target` / `riichi/Cargo.toml`；
- `riichi_ppo_v1`、`riichi_lab_bot` 中所有 `import riichi` 的安装说明；
- `.gitignore`、构建脚本、文档中的路径。

### 8.4 移动后验证

```bash
python -c "import riichi; print(riichi.__file__)"
python -c "import riichi; assert riichi.ANALYSIS_VERSION == 4"
pytest riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py \
  riichi_ppo_v1/tests/protocol/test_protocol_matrix.py \
  riichi_ppo_v1/tests/unit/test_feature_schema_v13.py -q

# 证明 riichi 不依赖 riichienv
rg -n "from riichienv|import riichienv" RiichiEnv/riichi/ || true
```

## 9. ranked 保留要求

1. 不得删除 `RANKED_URL`、`run_ranked`、`ranked` 子命令、`--forever`、相关重连逻辑。
2. 不得删除现有 ranked 单元测试；如果测试会连接真实端点，必须改为 mock，否则保留现状。
3. 新增或保留一个测试，证明 ranked 单元测试不会发起真实网络连接。
4. 本任务所有执行命令不得包含 `ranked`、`--forever`、`/ws/ranked`。
5. 最终报告必须明确声明：ranked 功能保留，本次未连接 ranked 端点。

## 10. 全仓库重复逻辑收敛

### 10.1 查找重复

```bash
# 同名函数/类
rg -n "^def |^class |^pub fn |^pub struct " --glob '*.py' --glob '*.rs' \
  | sed -E 's/.*:(def|class|pub fn|pub struct) //' | sort | uniq -d

# 相似代码块（先做人工/脚本辅助）
rg -l "tile_id_to_mjai" --glob '*.py'
rg -l "snapshot_json" --glob '*.py'
rg -l "_meld_field" --glob '*.py'
```

### 10.2 收敛规则

- 每个功能只保留一个权威实现；
- 权威实现放在依赖方向允许的公共位置；
- 删除所有副本；
- 为收敛后的函数新增等价测试；
- 如果两个副本已漂移，先明确“正确行为”，再统一到正确版本；
- 如果两个副本分别属于不同契约（如 V11/V13），不得强行合并；应把 V11 逻辑隔离到 legacy 目录。

## 11. 测试要求

### 11.1 基线

修改前先记录：

```bash
timeout 60 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -c \
  "import riichi, riichienv, riichi_ppo_v1, riichi_lab_bot; print('runtime ok')"
timeout 60 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -c \
  "import riichi, riichienv; assert getattr(riichi, 'ANALYSIS_VERSION', None) == 4; assert getattr(riichienv, 'REPLAY_SEMANTICS_VERSION', None) == 1; print('runtime contract ok')"

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
```

当前基线：`502 passed, 3 skipped`。

### 11.2 每次删除后

```bash
# 受影响模块定向测试
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest <受影响测试> -q

# 全量回归
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
```

### 11.3 最终回归（不可省略）

```bash
# 1) 语义逐 token 校验与事件/状态编码
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_lab_bot/tests/test_bridge_semantics.py \
  riichi_lab_bot/tests/test_bridge_integration.py \
  riichi_ppo_v1/tests/unit/test_semantic_validation.py \
  riichi_ppo_v1/tests/unit/test_feature_schema_v13.py \
  riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py \
  riichi_ppo_v1/tests/protocol/test_protocol_matrix.py \
  riichi_ppo_v1/tests/integration/test_real_action_cases.py \
  riichi_ppo_v1/tests/integration/test_bridge_integration.py -q

# 2) bot 与训练 bridge 逐 token 等价
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_lab_bot/tests/test_bridge_integration.py -q

# 3) 候选 token 漂移 = 0
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
  riichi_lab_bot/tools/verify_candidate_token_drift.py \
  --model checkpoints/train_riichi_v13_sft/best_heuristic.pt

# 4) 128 局事件/动作覆盖
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-ppo-validate \
  --games 128 --seed 20260713 --max-steps 2500 --output riichi_ppo_v1_coverage.json

# 5) 本地三局 + CPU 冒烟
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot local \
  --games 3 --seed 20260730 --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot local \
  --games 1 --device cpu --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt

# 6) FP32/BF16 一致性
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_lab_bot/tests/test_checkpoint.py::test_fp32_and_bf16_inference_agree_on_l20 -q

# 7) 线上 validation（只允许 validate）
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot validate \
  --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt \
  --jsonl-log logs/validate_v13_cleanup_$(date +%Y%m%d_%H%M%S).jsonl
```

### 11.4 线上验收

```text
validation_passed: true
exit code: 0
model_actions == requests
fallback_actions == 0
withheld_actions == 0
rejected == 0
unparseable == 0
stale == 0
defaulted == 0
```

## 12. 死代码/死配置/死文档专项检查

### 12.1 死代码

```bash
# 未被任何测试或入口引用的工具
for f in riichi_ppo_v1/tools/*.py riichi_lab_bot/tools/*.py; do
  name=$(basename "$f" .py)
  if ! rg -q "$name" --glob '*.py' --glob '*.toml' --glob '*.sh' .; then
    echo "UNREFERENCED $f"
  fi
done

# 未被加载入口读取的配置
for f in riichi_ppo_v1/configs/*.yaml; do
  name=$(basename "$f")
  if ! rg -q "$name" riichi_ppo_v1 --glob '*.py' --glob '*.yaml' --glob '*.toml'; then
    echo "UNREFERENCED $f"
  fi
done
```

### 12.2 死文档

```bash
# 文档中引用不存在的路径
rg -o '`[^`]+`' --glob '*.md' . \
  | tr -d '`' | sort -u \
  | while read -r p; do
      case "$p" in
        *://*|*'${'*|*'$'*) continue;;
      esac
      if [ ! -e "$p" ]; then echo "MISSING DOC PATH: $p"; fi
    done
```

### 12.3 残留缓存

```bash
git ls-files | rg '(target/|__pycache__/|\.pytest_cache/|\.so$|\.whl$)' || true
find . -name '*.pyc' -not -path './.git/*' | head -100
```

## 13. 提交与回滚策略

### 13.1 提交规范

每个提交：

```text
cleanup(scope): 一句话说明

- 删除/移动/收敛了什么
- 证据文件路径
- 测试命令与结果摘要
```

示例：

```text
cleanup(RiichiEnv): remove unused riichienv-ml package

- rg references: 0
- removed riichienv-ml/
- updated pyproject.toml workspace members
- pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests: 502 passed
```

### 13.2 回滚

```bash
# 单提交回滚
git revert <commit>

# 多提交回滚前先确认
git log --oneline -10
```

禁止使用 `git reset --hard`；如必须回退，使用 `git revert` 或先与用户确认。

## 14. 验收标准

| 编号 | 验收项 |
| --- | --- |
| A1 | 全仓库完成逐文件审计，不存在“未判定”文件。 |
| A2 | 每个删除对象都有“引用扫描 → 同步修改 → 删除 → 回归通过”四段证据。 |
| A3 | `riichi/` 已移入 `RiichiEnv/riichi/`，`import riichi` 与既有 API 可用，安装/CI/文档路径已同步。 |
| A4 | `ranked` 功能完整保留，相关代码、URL、CLI、单元测试均未被删除。 |
| A5 | 全量测试、语义校验、bridge 等价、drift=0、128 局覆盖、本地三局、CPU 冒烟、FP32/BF16、线上 validation 全部通过。 |
| A6 | 工作树与 git 索引中不存在 target、__pycache__、.pytest_cache、*.pyc、*.so、*.whl。 |
| A7 | 全仓库不存在两份可能漂移的同一功能实现。 |
| A8 | 最终报告明确声明：ranked 功能保留，本次未连接 ranked 端点。 |

## 15. 报告要求

最终报告必须包含：

1. 清理前后文件数、目录数、代码行数、体积对比表；
2. 全仓库冗余清单及每个候选的最终判定；
3. 每个删除对象的证据包：
   - 删除前引用扫描输出；
   - 删除后复扫输出；
   - 定向测试输出；
   - 全量测试输出；
4. 每个移动/重构对象的证据包：
   - 移动前后 API 对比；
   - import 验证输出；
   - 路径/CI/文档同步清单；
   - 相关测试输出；
5. 重复逻辑收敛表：
   - 原实现位置；
   - 权威实现位置；
   - 等价测试输出；
6. 覆盖率报告（before/after）；
7. 被保留但“看起来冗余”的文件清单及理由；
8. 所有回归命令原文与关键输出；
9. ranked 保留声明与“未连接真实 ranked 端点”声明；
10. 未解决问题/挂起候选清单。
