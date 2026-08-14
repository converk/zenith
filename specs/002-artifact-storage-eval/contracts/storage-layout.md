# Contract: 产物存储布局(Storage Layout)

适用仓库全部组件(riichi_ppo_v1、riichi_lab_bot、RiichiEnv)的产物目录、日志
写入点与文档路径。任何违背本契约的路径即为 bug。

## 1. 日志目录契约

- 所有运行日志(json/txt/log)必须写入 `logs/<版本号>/`,其中 `<版本号>` 匹配
  `^v[0-9]+$`。
- 禁止在 `logs/` 根目录、`audit/reports/` 或仓库其他位置生成日志文件。
- 代码默认值禁止写入 `logs/` 根目录:`sft/audit.py --report` 默认 `None`
  (不落盘),调用方显式指定路径。
- 训练进程与 Ray/子进程日志经 `RAY_LOG_TO_STDERR=1` + 运行脚本
  `exec > >(tee logs/<版本号>/...) 2>&1` 收敛。
- 例外(非运行日志,不适用本契约):`checkpoints/.../tensorboard/` 事件与
  `metrics.jsonl`、`heuristic_evaluation.jsonl` 等训练指标制品。

## 2. audit 目录契约

- `audit/reports/<版本号>/` 是初始设计文档、实验报告、测试与验证脚本的唯一
  存放位置;版本目录只允许四个固定类型子目录:

  | 类型 | 目录 | 内容 |
  |------|------|------|
  | design | `audit/reports/<v>/design/` | 初始设计文档(*.md) |
  | report | `audit/reports/<v>/report/` | 实验报告、进度记录、运行快照 |
  | eval | `audit/reports/<v>/eval/` | 评测与验证输出(1v3 固定输出) |
  | scripts | `audit/reports/<v>/scripts/` | 测试与验证脚本(*.py、*.sh) |

- 运行日志不属于 audit;存量归位时移入 `logs/<版本号>/`。
- 存量归位只移动/重命名,禁止删除:`v13_sft_20260802`→`v13`、
  `v14_ppo_20260812`→`v14`、`v15_ppo_20260814`→`v15`。

## 3. checkpoint 目录契约

- 新 checkpoint 固定保存到 `checkpoints/train_riichi_<版本号>` 下,按阶段分子
  目录(样板:`train_riichi_v15/ppo`、`train_riichi_v15/sft/stage_a`、
  `train_riichi_v15/sft/stage_b`);每个 checkpoint 内含配置快照。
- 存量目录只归档移动:
  `checkpoints/train_riichi_ppo_v14`→`checkpoints/train_riichi_v14`;
  `checkpoints/train_riichi_v13_sft`→`checkpoints/train_riichi_v13/sft`
  (若引用无法全部同步且测试通过,则作为存量例外保留原名)。
- checkpoint 禁止删除(含移动失败时的回滚,不得用删除代替)。
- 代码默认 checkpoint 目录必须中性:`checkpoints/train_riichi_current`;
  具体版本路径只出现在该版本的配置中。

## 4. 数据集目录契约

- 现行数据集:`datasets/tenhou_sft_2024_2025`(原始)与
  `datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16`(编码 40%)。
- 废弃数据集:`datasets/tenhou_sft_2024_2025_encoded_remaining_80pct_v11` 与
  `datasets/tenhou-to-mjai` 必须不存在(若存在,须经用户二次确认后删除)。
- 代码默认值不得指向废弃目录:`sft/prepare.py --archive-dir` 必填,无默认。

## 5. .gitignore 契约(方案 A,用户已确认)

```gitignore
audit/*
!audit/reports/
audit/reports/*
!audit/reports/*/
audit/reports/*/*
!audit/reports/*/design/
!audit/reports/*/report/
!audit/reports/*/scripts/
```

- `design/`、`report/`、`scripts/` 被 git 跟踪;`eval/` 与版本目录根散落文件
  保持忽略。
- 实现后用 `git check-ignore` 验证:design/report/scripts 下文件 NOT ignored,
  eval 下文件 IGNORED。

## 6. 文档一致性契约

- README/docs/AGENTS.md 中出现的每个数据集、checkpoint、日志、audit 路径必须
  真实存在且符合本契约;示例日志路径必须使用 `logs/<版本号>/`。
- CLI 默认值不得锁定历史版本(如 `v13_sft`、`v14`、`80pct_v11`、日期种子)。
