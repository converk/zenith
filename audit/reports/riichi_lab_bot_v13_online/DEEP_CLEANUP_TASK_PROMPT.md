# 任务：全仓库文件内部死代码与遗留逻辑深度清理（详细执行版）

> 本提示词是 `CLEANUP_TASK_PROMPT.md` 的增强版。`CLEANUP_TASK_PROMPT.md` 负责“文件/目录是否还需要”，本提示词负责“保留文件内部是否还有死代码、旧分支、重复逻辑、过时兼容层”。两份提示词的范围都是整个仓库，示例文件不是范围限制。

## 1. 目标

对全仓库每个保留文件，逐函数、逐类、逐常量、逐参数、逐分支、逐配置项进行审计，删除多轮迭代后积累的内部死代码，同时：

- 保持 V13 输入契约不变；
- 保持 `riichi.ANALYSIS_VERSION == 4`、`riichienv.REPLAY_SEMANTICS_VERSION == 1` 不变；
- 保留 ranked 功能；
- 不删除输入资产；
- 每个删除都有可复核的证据。

## 2. 与文件级清理的关系

```text
CLEANUP_TASK_PROMPT.md
  └─ 判断：文件/目录是否保留
        ├─ 删除 → 删除流程
        ├─ 移动 → 移动流程
        └─ 保留 → 进入本提示词
                    └─ 判断：文件内部哪些符号/分支/配置是死代码
                          ├─ 删除符号/分支/配置
                          ├─ 收敛重复实现
                          ├─ 提取到更合适位置
                          └─ 保留并说明理由
```

## 3. 内部审计对象

每个保留文件都必须回答：

```text
1. 这个文件的职责是什么？
2. 文件中的每个公开/私有符号分别属于什么职责？
3. 每个符号是否被当前代码、测试、配置、文档、CI、动态调用引用？
4. 每个分支/参数是否真的可能被当前行为触发？
5. 是否与仓库中其他文件重复？
6. 是否包含历史兼容层？当前是否还有兼容对象？
7. 删除后是否影响冻结契约或 ranked？
8. 如果不确定，证据是什么？
```

## 4. 内部审计表模板

对每个文件建立下表：

```markdown
## 文件：<路径>

职责：<一句话>
当前是否被运行路径引用：<是/否，证据>

| 符号/分支/配置 | 行号 | 类型 | 引用点 | 测试覆盖 | 判定 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `legacy_fixed` 分支 | 280-290 | 分支 | ... | ... | KEEP/DELETE | ... |
| `_legacy_v11` 参数 | 786 | 参数 | ... | ... | ... | ... |
| `RANKED_URL` | 22 | 常量 | client/cli/tests | ... | KEEP | ranked 功能保留 |
```

判定取值：

```text
KEEP                保留
KEEP_COMPAT          保留，因为属于冻结契约/公开 API/ranked
DELETE               删除
INLINE              参数/分支只有一个值，内联后删除
EXTRACT             从当前文件移到更合适的公共位置
MERGE               与另一处实现收敛
UNCERTAIN           挂起并报告
```

## 5. 死代码识别启发式

出现以下任一模式时，标记为“候选死代码”，但不直接删除：

### 5.1 未使用导入

```python
import os          # 从未使用
from typing import Any  # 从未使用
```

检查：

```bash
ruff check --select F401 <file>
python -m pyflakes <file> 2>/dev/null || true
```

### 5.2 未使用函数/类/常量

```bash
rg -n "^def |^class |^[A-Z_]+ =" <file>
for symbol in ...; do
  count=$(rg -n "$symbol" --glob '*.py' --glob '*.rs' --glob '*.yaml' \
    --glob '*.toml' --glob '*.md' --glob '*.sh' . | wc -l)
  echo "$symbol $count"
done
```

注意：定义处本身算一次引用；`count == 1` 才进入候选。

### 5.3 只有一个调用点的函数

```bash
rg -n "函数名" --glob '*.py' --glob '*.rs' .
```

如果只有一个调用点：

- 检查调用点是否生产路径；
- 如果函数逻辑简单，考虑内联；
- 如果函数逻辑复杂，考虑是否应移动为私有方法；
- 如果调用点本身也是死代码，先处理调用点。

### 5.4 永远为同一值的参数

```bash
rg -n "参数名" --glob '*.py' .
```

示例：

```python
def encode(observation, legacy=False):
    ...

# 所有生产调用都是 encode(obs, legacy=False)
```

处理：

- 删除参数；
- 内联固定值；
- 如果旧值由测试或旧 checkpoint 使用，先审计测试/资产；
- 如果参数属于 V11 兼容，按兼容策略处理。

### 5.5 永远为真的条件

```python
if True:
    ...

if legacy is False and condition:
    ...
```

检查：

- 条件中的变量是否只被赋一个值；
- 条件是否只在测试中为另一值；
- 如果是，删除分支并同步测试。

### 5.6 不可达代码

```python
return result
print("never")          # 不可达
raise RuntimeError()    # 后续代码不可达
```

检查：

- `return`、`raise`、`break`、`continue` 之后的语句；
- `while False`、`if False`；
- Rust 中 `unreachable!()` 之后或 `return` 之后的语句。

### 5.7 旧兼容层

常见模式：

```python
if contract is None and token_schema_version == 13:
    ...

if payload.get("sft_contract_version") is None:
    ...

_FORMAL_V13_MANIFEST_CONTRACT = (...)
```

处理：

- 先确认当前 `checkpoints/`、`datasets/` 是否还有这种格式；
- 没有 → 删除兼容层和对应测试；
- 有 → 保留，并在报告中列出资产路径；
- 无法确认 → 标记 UNCERTAIN。

### 5.8 旧版本分支（如 V11）

```python
if _legacy_v11:
    ...
else:
    ...
```

处理：

- 如果项目仍声明支持 V11：保留，但尽量把 V11 专属代码移入 `legacy/`；
- 如果项目不再支持 V11：删除 V11 分支、相关参数、相关测试、相关文档；
- 删除前必须确认 `legacy/v11` 适配器、V11 checkpoint、V11 测试都没有引用；
- 如果删除会影响 V13 路径，禁止删除。

### 5.9 ranked 相关代码

```python
RANKED_URL = "wss://game.riichi.dev/ws/ranked"

async def run_ranked(...):
    ...
```

处理：

- **保留**；
- 本任务不运行；
- 只允许清理 ranked 内部真正无引用的 helper，且必须保留功能；
- 删除任何 ranked 内部符号前，必须先跑 ranked 单元测试。

### 5.10 重复实现

```bash
# 同名符号
rg -n "^def |^class |^pub fn |^pub struct " --glob '*.py' --glob '*.rs' \
  | sed -E 's/.*:(def|class|pub fn|pub struct) //' | sort | uniq -d

# 已知重复示例（不是范围限制）
rg -l "tile_id_to_mjai" --glob '*.py'
rg -l "snapshot_json" --glob '*.py'
rg -l "_meld_field" --glob '*.py'
```

处理：

- 选择权威实现；
- 删除副本；
- 新增等价测试；
- 如果两份已漂移，先确定正确行为，再统一；
- 如果属于不同契约，不强行合并，隔离到各自目录。

## 6. 证据包要求

每个删除/收敛对象必须形成证据包：

```markdown
### 候选：<符号/分支/配置>

位置：<文件:行号>
类型：<函数/类/常量/参数/分支/配置 key>

引用扫描：
```bash
rg -n "<符号>" ...   # 输出
```

动态引用扫描：
```bash
rg -n '"<符号>"' --glob '*.py' --glob '*.yaml' --glob '*.toml' ...
```

测试覆盖：
```bash
coverage report --show-missing  # 相关行是否被覆盖
```

当前运行路径是否依赖：<是/否>
是否属于冻结契约：<是/否>
是否属于 ranked：<是/否>

判定：<DELETE/INLINE/EXTRACT/MERGE/KEEP/UNCERTAIN>
理由：<一段话>

删除/修改后测试：
```bash
pytest <相关测试> -q
pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
```
```

## 7. 执行顺序

```text
Step 1: 建立全仓库文件清单
Step 2: 对每个保留文件建立内部审计表
Step 3: 生成候选死代码列表（引用扫描 + 覆盖率 + 静态检查）
Step 4: 对每个候选收集证据包
Step 5: 按“先低风险后高风险”顺序处理：
        未使用导入
        → 未使用私有函数
        → 永远同一值的参数
        → 重复实现收敛
        → 旧兼容层
        → 旧版本分支
        → 公开 API/跨模块重构
Step 6: 每个处理后运行定向测试 + 全量测试
Step 7: 最终回归阶梯
Step 8: 写报告
```

## 8. 禁止事项

- 禁止“先删后查”；
- 禁止只依据“看起来没用”删除；
- 禁止删除 ranked；
- 禁止修改 V13 冻结契约；
- 禁止删除输入资产；
- 禁止把不确定候选静默保留或静默删除；
- 禁止为了覆盖率数字删除必要边界处理；
- 禁止在同一提交里混合“删除死代码”和“功能重构”；
- 禁止在未跑全量测试前声称完成。

## 9. 测试要求

### 9.1 覆盖率

```bash
cd /mnt/disk1/hubowen/zenith
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pip show coverage >/dev/null 2>&1 || \
  /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pip install coverage

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m coverage run \
  --source=riichi_ppo_v1,riichi_lab_bot,RiichiEnv/src/riichienv,RiichiEnv/riichi \
  -m pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m coverage report --show-missing
```

对每个 `Missing` 行：

- 如果无法被任何测试覆盖且无引用 → 候选删除；
- 如果属于难以构造的边界（如 chankan、多家和、deadline）→ 保留，并在报告中说明；
- 如果属于 ranked → 保留，但可由单元测试覆盖 mock 路径；
- 如果属于 V13 冻结契约 → 保留，必须有对应测试。

### 9.2 静态检查

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m ruff check \
  riichi_ppo_v1 riichi_lab_bot RiichiEnv/src/riichienv

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m compileall -q \
  riichi_ppo_v1 riichi_lab_bot RiichiEnv/src/riichienv
```

### 9.3 删除后反向证明

```bash
rg -n "被删除符号" \
  --glob '!**/target/**' --glob '!**/__pycache__/**' \
  --glob '!checkpoints/**' --glob '!datasets/**' --glob '!logs/**' . || true
```

输出必须为 0。

### 9.4 等价测试

对重复收敛：

- `tile_id_to_mjai`：遍历 0–135 全部 tile id；
- `_normalized_action_json`：覆盖 `dahai/chi/pon/daiminkan/none`；
- `snapshot_json`：覆盖本地与线上缺失字段 Observation；
- public summary：bot 与训练端逐 token 相等；
- 其他收敛函数：用删除前实现作为 golden。

### 9.5 最终回归（与 `CLEANUP_TASK_PROMPT.md` 一致）

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest \
  riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q

CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
  riichi_lab_bot/tools/verify_candidate_token_drift.py \
  --model checkpoints/train_riichi_v13_sft/best_heuristic.pt

CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-ppo-validate \
  --games 128 --seed 20260713 --max-steps 2500 --output riichi_ppo_v1_coverage.json

CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot local \
  --games 3 --seed 20260730 --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt

CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot validate \
  --device cuda:0 --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt \
  --jsonl-log logs/validate_v13_deep_cleanup_$(date +%Y%m%d_%H%M%S).jsonl
```

## 10. 验收标准

| 编号 | 验收项 |
| --- | --- |
| D1 | 全仓库每个保留文件都有内部审计表。 |
| D2 | 每个删除符号都有完整证据包。 |
| D3 | 删除后全仓库引用扫描为 0。 |
| D4 | 重复实现已收敛，不存在两份可能漂移的同名实现。 |
| D5 | 旧兼容层/旧版本分支要么删除并同步删测试，要么明确保留并说明理由。 |
| D6 | ranked 功能完整保留，相关代码、URL、CLI、单元测试均未被删除。 |
| D7 | 全量测试、语义校验、bridge 等价、drift=0、128 局覆盖、本地三局、CPU 冒烟、FP32/BF16、线上 validation 全部通过。 |
| D8 | 最终报告明确声明：ranked 功能保留，本次未连接 ranked 端点。 |

## 11. 报告要求

最终报告必须包含：

1. 全仓库逐文件内部审计表；
2. 候选死代码总清单及判定；
3. 每个删除/收敛符号的证据包；
4. 覆盖率报告（before/after）；
5. 静态检查输出；
6. 等价测试输出；
7. 全量回归命令与输出；
8. 被保留的兼容层/旧分支清单及理由；
9. ranked 保留声明；
10. 挂起/不确定候选清单及等待用户决策的问题。
