# Contract: CLI 与配置契约(治理后)

本文档定义治理完成后的对外命令行与配置契约,是实现的验收面。治理不改变已发布的
训练/环境算法语义,只收紧入口与默认值。

## riichi-ppo-train / riichi-ppo-smoke(riichi_ppo_v1.training.train)

### 配置加载

- `--config <path>`:指向自包含版本配置;直接加载,不与打包默认叠加。
- 未提供 `--config`:使用打包默认(`configs/training.yaml` +
  `configs/monitoring.yaml` 合并)。
- 移除 `--model-config`、`--environment-config`、`--training-config`。
- 显式 CLI 值覆盖(`--device`、`--iterations`、`--checkpoint-dir`、
  `--num-workers`、`--learner-gpus`、`--envs-per-worker`、`--kyokus-per-worker`、
  `--update-epochs`、`--minibatch-size`、`--target-kl` 等)仍优先生效。

### 1v3 评测键(版本配置内)

| 键 | 语义 | 默认 |
|----|------|------|
| `eval1v3_enabled` | 是否启用 | false |
| `eval1v3_interval_updates` | 固定机制常量 30 | 常量 |
| `eval1v3_processes` | 固定机制常量 10 | 常量 |
| `eval1v3_hanchans_per_process` | 固定机制常量 160 | 常量 |
| `eval1v3_model_b` | 对手模型路径 | 必填(开启时) |
| `eval1v3_seed_base` | 种子基数 | 必填(开启时) |
| `eval1v3_devices` | 评测设备 | 缺省 ("0","1") |
| `eval1v3_output_dir` | 结果目录 | 必填(开启时) |

旧 `evaluation_*` 键全部移除;`evaluation.jsonl` 不再产生。

## riichi-ppo-validate(riichi_ppo_v1.tools.validate)

- 入口从 `riichi_ppo_v1.tests.validate:main` 迁至
  `riichi_ppo_v1.tools.validate:main`。
- `--games` 默认 128;`--seed` 默认 0(中性,不再 20260713);
  `--max-steps` 默认 2500;`--output` 默认 `riichi_ppo_v1_coverage.json`。

## riichi-sft-prepare / riichi-sft-precompute

- `riichi-sft-prepare --archive-dir` 默认 `datasets/tenhou_sft_2024_2025`
  (现行数据集,原则 III 认可)。
- `riichi-sft-precompute --source` 默认现行数据集;`--output` 必填,不再默认锁定
  历史编码名。

## riichi-lab-bot(riichi_lab_bot.cli)

- `--checkpoint`:`RIICHI_CHECKPOINT` 环境变量或显式 `--checkpoint` 必填,
  不再硬编码 `checkpoints/train_riichi_ppo_v14/checkpoint_00510.pt`。
- `local --seed` 默认 0(不再 20260730);`--games`/`--max-steps` 保持 3/4000。

## 领域常量

- Python 单一来源 `riichi_ppo_v1.model.schema`: `NUM_ACTIONS=241`、
  `TILE_KINDS=34`、`TOKEN_SCHEMA_VERSION=13`。
- Rust:`riichienv-state-machine` `NUM_ACTIONS=241`;`riichienv-core`
  `TILES_4P=136`。
- `RiichiEnv/src/riichienv/convert.py`:`TID_COUNT=136`。

## 协议边界(不可变)

- `riichienv-state-machine` 公开模块名保持 `riichi`,不依赖 `riichienv`;
  `RiichiEnv` 的 MJAI 状态机协议与 `KyokuEventTupleProtocol.md` /
  `KyokuActionSpace.md` 保持兼容。
