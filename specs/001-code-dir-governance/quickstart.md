# Quickstart: 治理验证指南

以下命令从仓库根执行,Conda 环境 `Mahjong-AI`。

## 1. 快速单元验证(每个治理主题后必跑)

```bash
conda run -n Mahjong-AI python -m pytest riichi_ppo_v1/tests/unit \
  riichi_ppo_v1/tests/protocol -q
conda run -n Mahjong-AI python -m pytest riichi_lab_bot/tests -q
conda run -n Mahjong-AI python -m pytest RiichiEnv/tests -q
```

## 2. Rust 侧验证

```bash
cd RiichiEnv/riichienv-core && cargo test
cd ../../RiichiEnv/riichienv-state-machine && cargo test
cd ../..
```

## 3. 配置契约验证

```bash
conda run -n Mahjong-AI python - <<'PY'
from riichi_ppo_v1.training.train import load_config

defaults = load_config()
assert defaults["policy_head_type"] == "isolated_action_query"
assert "evaluation_enabled" not in defaults
assert defaults["checkpoint_dir"] == "checkpoints/train_riichi_current"

v15 = load_config("riichi_ppo_v1/configs/v15_ppo.yaml")
assert v15["policy_head_type"] == "isolated_action_query"
assert v15["num_workers"] == 12          # 自包含,不依赖打包默认
assert v15["eval1v3_model_b"]            # 1v3 对手来自配置

v14_resume = load_config("riichi_ppo_v1/configs/v14_ppo_resume.yaml")
assert v14_resume["resume"] == "checkpoints/train_riichi_ppo_v14/checkpoint_00600.pt"
assert v14_resume["init_model"] is None
PY
```

## 4. 幽灵引用零命中

```bash
rg -n "legacy/v11|legacy_fixed|build_legacy_v11|exp/|train_riichi_ppo\b|encoded_10pct" \
  AGENTS.md riichi_ppo_v1 riichi_lab_bot RiichiEnv docs \
  --glob '!*.pyc' --glob '!target/**'
# 期望: 无输出(或仅命中小节/历史说明中的明确豁免,实现时逐条核对)
```

## 5. 冒烟清理验证

```bash
conda run -n Mahjong-AI riichi-ppo-smoke --device cpu
test ! -d checkpoints/riichi_ppo_v1_smoke && echo "smoke cleaned"
```

## 6. 入口与打包验证

```bash
conda run -n Mahjong-AI python -m pip install -e riichi_ppo_v1 --no-deps --no-build-isolation -q
conda run -n Mahjong-AI riichi-ppo-validate --help
conda run -n Mahjong-AI python -c "import riichi_ppo_v1.tests" 2>&1 | tail -1
# 期望: tests 不再被打包/暴露为运行时包
```

## 7. 目录职责清单

```bash
test -f docs/directory-responsibilities.md && \
rg -n "riichi_ppo_v1/(evaluation|model|sft|training|tools)|riichi_lab_bot|RiichiEnv" \
  docs/directory-responsibilities.md
```

## 8. 性能/训练基线(需要 GPU,验收时按宪法基线)

```bash
CUDA_DEVICE=0,1 conda run -n Mahjong-AI riichi-ppo-train \
  --config riichi_ppo_v1/configs/v15_ppo.yaml \
  --device cuda --learner-gpus 2 --iterations 3 \
  --num-workers 12 --envs-per-worker 32 --kyokus-per-worker 16 \
  --update-epochs 4 --minibatch-size 512 --target-kl 0.0 \
  --checkpoint-dir checkpoints/train_riichi_benchmark
```

跑 3 轮,第 1 轮为预热,单独报告第 2–3 轮的耗时与指标;结束后清理
`checkpoints/train_riichi_benchmark` 与本轮日志。
