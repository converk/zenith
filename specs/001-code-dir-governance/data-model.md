# Data Model: 代码与目录治理

本 feature 不新增运行时数据实体;以下是治理过程涉及的抽象及其一致性规则,供
tasks 与验收引用。

## 实体

### 配置单元(Configuration Unit)

- **定义**: 一个可被入口直接加载、无需叠加其它文件的 YAML 配置。
- **字段**:
  - `path`: 文件路径,如 `riichi_ppo_v1/configs/v15_ppo.yaml`;
  - `self_contained`: 布尔,加载器要求版本配置为真;
  - `kind`: `packaged_default`(training.yaml/monitoring.yaml 组合)或
    `version_config`(v14/v15/resume/sft);
  - `keys`: 运行所需的完整键集合。
- **校验规则**:
  - `kind=version_config` 的配置必须自包含;`v14_ppo_resume.yaml` 必须与
    `v14_ppo.yaml` 内容一致(除 `resume`/`init_model`);
  - `packaged_default` 不得含旧 `evaluation_*` 键;
  - 1v3 开启(`eval1v3_enabled=true`)时 `eval1v3_model_b`、`eval1v3_seed_base`、
    `eval1v3_output_dir` 必填;
  - 默认 `checkpoint_dir` 为中性路径 `checkpoints/train_riichi_current`。
- **关系**: 每个治理主题(见下)可修改 0..N 个配置单元;测试用例断言其
  不变量。

### 治理主题(Governance Topic)

- **定义**: 一个 commit 对应的独立变更单元,主题范围与依赖顺序见
  research.md Decision 11。
- **字段**: `topic_id`、`title`、`affected_files`、`verification`(对应测试命令)。
- **校验规则**: 每主题 commit 前跑对应测试;删除类主题先 `rg` 全仓库零引用。

### 契约与领域常量(Contract & Domain Constants)

- **定义**: 单一来源的契约 ID 与领域不变常量。
- **字段**:
  - Python 侧: `TOKEN_SCHEMA_VERSION=13`(已有,`model/schema.py`)、
    `NUM_ACTIONS=241`、`TILE_KINDS=34`(新增同文件);
  - Rust 侧: `riichienv-state-machine` `NUM_ACTIONS=241`(已有)、
    `riichienv-core` `TILES_4P=136`(已有);
  - `RiichiEnv/src/riichienv/convert.py`: `TID_COUNT=136`(新增)。
- **校验规则**:
  - 三组件内各常量只有一处定义,其余位置引用;
  - `architecture.py`/`bridge.py` 不再各自定义 `NUM_ACTIONS`;
  - 跨语言数值一致性由既有协议测试(`test_action_space_exhaustive.py`、
    Rust 语义测试)覆盖。

### 目录职责条目(Directory Responsibility Entry)

- **定义**: `docs/directory-responsibilities.md` 中的一行:目录路径 + 一句话职责。
- **字段**: `directory`、`responsibility`。
- **校验规则**: 覆盖三组件每个源码/配置目录;与治理后实际结构一致。
