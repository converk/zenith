#!/usr/bin/env python
"""V19 显存实测：B=2048 单次 full forward+backward 峰值。

验收线（AGENTS/实施计划阶段7）：B=2048 峰值 allocated ≤35GB；超线预案为
actor 单 block 梯度检查点。本脚本只做合成输入的前向/反向，不启动训练。

运行（使用单卡 CUDA_VISIBLE_DEVICES=0）：
  CUDA_VISIBLE_DEVICES=0 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python \
    audit/reports/v19/scripts/measure_v19_memory.py --batch 2048
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from riichi_ppo_v1.model import KyokuTransformerActorCritic, ModelConfig
from riichi_ppo_v1.model.belief_network import BeliefNetwork
from riichi_ppo_v1.tests.v19_fixtures import actor_inputs, critic_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    torch.manual_seed(2026)
    model = KyokuTransformerActorCritic(ModelConfig.preset("v19")).to(device)
    params = sum(p.numel() for p in model.parameters())
    inputs = actor_inputs(batch=args.batch, action_ids=(1, 7, 12))
    critic = critic_inputs(batch=args.batch)
    gpu_inputs = {
        "actor_factors": inputs["actor_factors"].to(device),
        "actor_numeric": inputs["actor_numeric"].to(device),
        "actor_lengths": inputs["actor_lengths"].to(device),
        "query_action_ids": inputs["action_ids"].to(device),
        "query_pair_counts": inputs["query_pair_counts"].to(device),
        "legal_mask": inputs["legal_mask"].to(device),
        "critic_factors": critic["critic_factors"].to(device),
        "critic_lengths": critic["critic_lengths"].to(device),
        "belief_public_grad_scale": 0.25,
    }
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = model(**gpu_inputs)
    output["policy_logits"].float().sum().backward()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    report = {
        "batch": args.batch,
        "device": str(device),
        "params": params,
        "peak_allocated_gb": round(float(peak) / 1024**3, 3),
        "peak_reserved_gb": round(float(reserved) / 1024**3, 3),
        "elapsed_s": round(float(elapsed), 3),
        "acceptance_limit_gb": 35.0,
        "pass": float(peak) / 1024**3 <= 35.0,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
