# Quickstart: 产物存储与评测机制固化的验证指南

在实现完成后,用以下场景端到端验证 feature。破坏性操作(删除 `logs/` 存量、删除
数据集)在执行前必须先向用户确认;本指南不含删除命令本身,只验证规范是否生效。

## 前置条件

```bash
conda activate Mahjong-AI
cd /mnt/disk1/hubowen/zenith
python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation
```

## 场景 1:日志与 audit 目录规范

```bash
# logs/ 根目录必须为空
ls -A logs/           # 期望:无输出(或只有版本子目录)
find logs -maxdepth 1 -type f   # 期望:0 个文件

# audit 顶层只允许三个版本目录,内部只有固定类型子目录
ls audit/reports/     # 期望:v13 v14 v15
find audit/reports/v15 -mindepth 1 -maxdepth 1
# 期望:design/ report/ eval/ scripts/(无其它散落文件)

# .gitignore 放行:design/report/scripts 入库,eval 忽略
git check-ignore audit/reports/v15/design/x.md && echo BAD || echo OK
git check-ignore audit/reports/v15/eval/x.json && echo OK || echo BAD
```

## 场景 2:checkpoint 布局与零删除

```bash
ls checkpoints/       # 期望:train_riichi_v13 train_riichi_v14 train_riichi_v15
test -d checkpoints/train_riichi_v13/sft
test -f checkpoints/train_riichi_v13/sft/best_heuristic.pt
# 旧路径零引用
rg -n 'train_riichi_ppo_v14|train_riichi_v13_sft' riichi_ppo_v1 riichi_lab_bot \
  --glob '!*.pyc' --glob '!*.pt' --glob '!*.jsonl' --glob '!*.log' || echo PASS
```

## 场景 3:数据集规范

```bash
ls datasets/
# 期望:tenhou_sft_2024_2025 与 tenhou_sft_2024_2025_encoded_40pct_v13_v16
test ! -e datasets/tenhou-to-mjai
test ! -e datasets/tenhou_sft_2024_2025_encoded_remaining_80pct_v11
python -m riichi_ppo_v1.sft.prepare --help | grep -A1 archive-dir
# 期望:--archive-dir 无默认值(required)
```

## 场景 4:1v3 机制与输出

```bash
python -m riichi_ppo_v1.evaluation.head_to_head_1v3 --help
# 期望:--hanchans 默认 1600、--parallel-hanchans 默认 160、--seed-base 默认 0

python - <<'PY'
from riichi_ppo_v1.evaluation.mechanism import (
    REQUIRED_1V3_PROCESSES, DEFAULT_1V3_HANCHANS_PER_PROCESS,
    TOTAL_1V3_HANCHANS, DEFAULT_1V3_INTERVAL_UPDATES,
)
assert (REQUIRED_1V3_PROCESSES, DEFAULT_1V3_HANCHANS_PER_PROCESS) == (10, 160)
assert TOTAL_1V3_HANCHANS == 1600 and DEFAULT_1V3_INTERVAL_UPDATES == 30
print("PASS")
PY

grep 'eval1v3_output_dir' riichi_ppo_v1/configs/v14_ppo.yaml \
  riichi_ppo_v1/configs/v15_ppo.yaml
# 期望:audit/reports/v14/eval 与 audit/reports/v15/eval
```

## 场景 5:SFT 节奏单点

```bash
grep -E 'interval_steps|hanchan_count' riichi_ppo_v1/configs/sft.yaml
# 期望:3000 与 96;实验配置无这些键
rg -n 'interval_steps|hanchan_count' riichi_ppo_v1/configs/v15_sft_*.yaml \
  && echo BAD || echo PASS
```

## 场景 6:测试与交付

```bash
python -m pytest -q
# 期望:三组件 Python 测试全通过
cargo test --manifest-path RiichiEnv/riichienv-state-machine/Cargo.toml
cargo test --manifest-path RiichiEnv/riichienv-core/Cargo.toml
# 期望:Rust 测试通过
git log --oneline -10
# 期望:本 feature 按主题切分为多个 commit
```

## 预期结果汇总

- `logs/` 根目录空,新日志只出现在 `logs/<版本号>/`;
- `audit/reports/{v13,v14,v15}` 各含 `design/ report/ eval/ scripts/`;
- `checkpoints/` 为 `train_riichi_{v13,v14,v15}`,无旧路径引用;
- 废弃数据集不存在、`prepare.py --archive-dir` 必填;
- 1v3 常量单一来源且输出固定 `audit/reports/<版本号>/eval`;
- SFT 节奏只在 `sft.yaml`(及契约常量)定义;
- 全量测试通过,交付按主题分 commit。
