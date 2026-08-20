# Quickstart: RiichiLab Bot V16 输入适配

## Prerequisites

```bash
conda activate Mahjong-AI
python -c "import riichi, riichienv, riichi_ppo_v1, riichi_lab_bot; print('runtime ok')"
```

## Unit and Semantic Tests

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest -q \
  riichi_ppo_v1/tests/unit/test_semantic_validation.py \
  riichi_ppo_v1/tests/integration/test_v16_encoding_bridge.py \
  riichi_ppo_v1/tests/integration/test_v16_query_semantics.py

/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest -q \
  riichi_lab_bot/tests
```

Expected outcome: V16 semantic tests pass; bot checkpoint, bridge, client and safety tests pass.

## Local Bot Run

```bash
CUDA_DEVICE=0,1 riichi-lab-bot local \
  --games 3 \
  --seed 20260730 \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/latest.pt
```

Expected outcome: three games complete; first is warmup; measured games report decisions/s; fallback and withheld counts are zero.

## Online Validation Only

```bash
mkdir -p logs/v17
RIICHI_BOT_TOKEN=<provided-token> CUDA_DEVICE=0,1 riichi-lab-bot validate \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v17/archive_20260819_V1run1/ppo/latest.pt \
  --jsonl-log logs/v17/bot-validate.jsonl
```

Expected outcome: validation exits 0 only after RiichiLab reports passed. Do not run `ranked` for this feature.
