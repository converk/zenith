# Contract: 固定评测机制(Evaluation Mechanism)

PPO 唯一评测机制为固定 1v3 对抗;SFT 采用固定 3000 steps 节奏。机制常量的修改
必须走宪法修订,禁止在实验配置或 CLI 默认值中悄悄改变。

## 1. 1v3 机制常量(单一来源)

定义于 `riichi_ppo_v1/evaluation/mechanism.py`,其余模块只可导入引用:

| 常量 | 值 | 含义 |
|------|----|------|
| `REQUIRED_1V3_PROCESSES` | 10 | 固定进程数 |
| `DEFAULT_1V3_HANCHANS_PER_PROCESS` | 160 | 每进程半庄数 |
| `TOTAL_1V3_HANCHANS` | 1600 | 总半庄数(10 × 160) |
| `DEFAULT_1V3_INTERVAL_UPDATES` | 30 | 每 30 updates 评测一次 |

- `training/train.py`、`evaluation/head_to_head_1v3_shards.py`、
  `evaluation/head_to_head_1v3.py` 全部从 `mechanism.py` 导入;
  `head_to_head_1v3.py` 的 CLI 默认值必须一致:`--hanchans` 默认 1600、
  `--parallel-hanchans` 默认 160、`--seed-base` 默认 0(中性)。
- 评测必须恰好 10 进程、每进程 160 半庄;`run()` 与
  `run_sharded_1v3()` 对进程数不符必须报错。

## 2. 可变参数(配置/CLI 指定,代码无版本默认)

当 `eval1v3_enabled: true` 时,以下键必填(缺失即报错),默认值不得锁定任何历史
版本:

| 键 | 必填 | 说明 |
|----|------|------|
| `eval1v3_model_b` | 是 | 对手模型路径 |
| `eval1v3_seed_base` | 是 | 种子基数 |
| `eval1v3_output_dir` | 是 | 必须为 `audit/reports/<版本号>/eval` |
| `eval1v3_devices` | 否 | 缺省中性 `("0","1")`,数量必须整除进程数 |
| `eval1v3_parallel_hanchans` | 否 | 缺省等于 `DEFAULT_1V3_HANCHANS_PER_PROCESS` |

## 3. 输出落点

- 分片与汇总:`audit/reports/<版本号>/eval/vs_sft_u<update:03d>.json` 与
  `.../eval/shards/`;
- 评测摘要轮转:`audit/reports/<版本号>/eval/eval1v3.jsonl`(从
  `checkpoint_dir/eval1v3.jsonl` 迁出);
- 进度报告:`audit/reports/<版本号>/report/PROGRESS.md`
  (`_progress_md_path` 由 `eval1v3_output_dir` 推导:`output_dir.parent/report/
  PROGRESS.md`;`eval1v3_output_dir` 缺省时跳过进度写入)。

## 4. SFT 节奏(单点定义)

- 机制源:`sft/contract.py` 的 `SFT_CADENCE_STEPS = 3000` 与
  `SFT_FINAL_EVAL_HANCHAN_COUNT = 96`。
- 唯一 YAML 载体:`configs/sft.yaml` 显式列出:
  `validation_interval_steps: 3000`、`checkpoint_interval_steps: 3000`、
  `heuristic_evaluation_interval_steps: 3000`、
  `heuristic_evaluation_hanchan_count: 96`、
  `heuristic_evaluation_final_hanchan_count: 96`。
- 实验配置(如 `v15_sft_offense_warmup.yaml`、`v15_sft_actor_finetune.yaml`)
  禁止包含上述任何节奏键;`DEFAULT_CONFIG` 以契约常量为中性兜底。
- 一致性测试断言 `sft.yaml` 数值 == 契约常量,且实验配置零复制。
