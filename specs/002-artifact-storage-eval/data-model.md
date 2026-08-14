# Data Model: 产物存储与评测机制固化

本 feature 不引入持久化数据库;以下实体描述的是目录布局、命名契约与配置键的
结构关系,作为实现与测试的判定依据。

## 实体

### 1. 版本号(VersionTag)

- **含义**: 一个代际标签(`v13`、`v14`、`v15`…),同时命名
  `logs/<版本号>/`、`audit/reports/<版本号>/` 与
  `checkpoints/train_riichi_<版本号>`。
- **字段**:
  - `tag`: 小写 `v` + 数字(如 `v15`),全仓库唯一的裸代际标签;
  - `checkpoint_dir`: `checkpoints/train_riichi_<tag>` 或其阶段子目录;
  - `log_dir`: `logs/<tag>/`;
  - `audit_dir`: `audit/reports/<tag>/`。
- **关系**: 一个 VersionTag 拥有一个 CheckpointLayout、一个 audit 目录与一个
  log 目录;多个 EvaluationRun 可属于同一 VersionTag。
- **校验规则**: `tag` 必须匹配 `^v[0-9]+$`;三个目录名由同一 `tag` 派生,禁止
  `v14_ppo_20260812` 一类混入日期/阶段的目录名。

### 2. 日志制品(LogArtifact)

- **含义**: 运行时产生的 json/txt/log 事件流(训练输出重定向、Ray 日志、bot
  事件 jsonl、audit 脚本的运行日志)。
- **字段**:
  - `path`: 必须位于 `logs/<版本号>/` 下;
  - `producer`: 组件(riichi_ppo_v1 / riichi_lab_bot / RiichiEnv)与入口;
  - `kind`: json/txt/log。
- **关系**: 属于一个 VersionTag 的 `log_dir`;与 AuditArtifact 互斥(audit 目录内
  不允许存在运行日志)。
- **校验规则**: 任何代码默认值、脚本重定向、文档示例写入的日志路径必须满足
  `logs/<版本号>/**`;`logs/` 根目录必须为空。

### 3. 审计制品类型(AuditArtifactType)

- **含义**: `audit/reports/<版本号>/` 下的固定四类内容。
- **字段**(固定枚举):
  - `design`: 初始设计文档(*.md);
  - `report`: 实验报告、进度记录与运行快照(REPORT.md、PROGRESS.md、
    environment/git_status/commands 等);
  - `eval`: 评测与验证输出(1v3 的 `vs_sft_u*.json`、`shards/`、历史评测目录);
  - `scripts`: 测试与验证脚本(*.py、*.sh)。
- **关系**: 四类分别对应 `audit/reports/<版本号>/design|report|eval|scripts` 目录;
  `design`、`report`、`scripts` 被 git 跟踪,`eval` 保持忽略。
- **校验规则**: 版本目录下除四个固定类型子目录外不得出现其它文件/目录;类型名
  固定,禁止随意命名。

### 4. Checkpoint 布局(CheckpointLayout)

- **含义**: 训练产物的目录组织:`checkpoints/train_riichi_<版本号>` 下按阶段分子
  目录(如 `sft/stage_a`、`sft/stage_b`、`ppo`)。
- **字段**:
  - `root`: `checkpoints/train_riichi_<tag>`;
  - `stage`: 阶段路径(`sft`、`sft/stage_a`、`ppo` 等);
  - `snapshot`: 每个 checkpoint 文件内含的配置快照键。
- **关系**: 属于一个 VersionTag;`init_model`、`resume`、`eval1v3_model_b` 等
  配置键引用该布局中的具体文件。
- **状态转换**:
  - 存量目录(`train_riichi_ppo_v14`、`train_riichi_v13_sft`)→ 归档移动为规范名
    (`train_riichi_v14`、`train_riichi_v13/sft`),只允许 `move`,不允许 `delete`;
  - 若移动后引用无法同步通过测试 → 回退为存量例外(保留原名)。
- **校验规则**: 新 checkpoint 目录必须为 `checkpoints/train_riichi_<版本号>`
  (或阶段子目录);代码默认 checkpoint 目录必须中性
  (`checkpoints/train_riichi_current`)。

### 5. 评测运行(EvaluationRun)

- **含义**: 一次固定 1v3 对抗评测。
- **字段**:
  - `processes`: 固定 10;
  - `hanchans_per_process`: 固定 160;
  - `total_hanchans`: 固定 1600;
  - `interval_updates`: 固定 30;
  - `model_b`: 对手模型路径(配置/CLI 指定,无版本默认);
  - `seed_base`、`devices`、`output_dir`: 版本配置指定;
  - `output_dir`: `audit/reports/<版本号>/eval`。
- **关系**: 属于一个 VersionTag;产出的 `vs_sft_u<update>.json`、`shards/`、
  `eval1v3.jsonl` 摘要落在 `output_dir`;`PROGRESS.md` 落在
  `audit/reports/<版本号>/report`。
- **校验规则**: 机制字段只能来自 `evaluation/mechanism.py` 常量;`model_b`、
  `seed_base`、`output_dir` 缺省时(1v3 开启)必须报错;`output_dir` 必须匹配
  `audit/reports/v[0-9]+/eval`。

### 6. SFT 节奏参数(SftCadence)

- **含义**: SFT 阶段验证、启发式评测与 checkpoint 保存的统一节奏。
- **字段**:
  - `interval_steps`: 3000(验证/启发式评测/checkpoint 保存共用);
  - `final_eval_hanchan_count`: 96;
  - `interval_eval_hanchan_count`: 96(与最终评估一致)。
- **关系**: 机制源为 `sft/contract.py` 的命名常量;`sft.yaml` 是唯一 YAML 载体;
  实验配置不得包含这些键。
- **校验规则**: `sft.yaml` 数值与契约常量一致(一致性测试);实验配置零复制;
  改动必须走宪法修订。

## 跨实体约束

- 版本号同源:`logs`、`audit/reports`、`checkpoints/train_riichi_` 三处目录名由
  同一个 `VersionTag.tag` 派生。
- 类型互斥:运行日志只进 `logs/<版本号>/`,不进入 `audit/reports/`;
  评测输出只进 `eval/` 类型,不进入 `logs/`。
- 只增不减:checkpoint 只归档移动;audit 存量归位只移动/重命名;删除仅限用户
  已确认的 `logs/` 存量与两个废弃数据集。
