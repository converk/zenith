"""GRP 训练入口契约测试:E[U] 辅助损失与最优权重重载冻结。

覆盖两个契约:
1. ``eu_loss_coef`` 损失项:>0 时训练梯度含 E[U] 回归项,=0 时退回纯 CE;
2. 训练结束的重载冻结语义:最终 val 差于中间最优时,落盘 best.pt 必须是
   中间最优权重且 validation_loss 标签一致(修复无条件覆盖 bug)。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from riichi_ppo_v1.model.grp import GRP_INPUT_SIZE, GRPModel
from riichi_ppo_v1.training.grp import train as grp_train


def _fake_rows(count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """构造确定性假样本:(T,21) 特征与 (4,) 最终排名(rank_by_player)。"""
    rng = np.random.default_rng(42)
    rows = []
    for index in range(count):
        length = 2 + index % 3
        features = rng.standard_normal((length, GRP_INPUT_SIZE)).astype(np.float32)
        ranks = np.array(
            [index % 4, (index + 1) % 4, (index + 2) % 4, (index + 3) % 4],
            dtype=np.int64,
        )
        rows.append((features, ranks))
    return rows


def _base_config(tmp_path: Path, eu_loss_coef: float) -> dict:
    return {
        "seed": 1,
        "device": "cpu",
        "epochs": 2,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "eu_loss_coef": eu_loss_coef,
        "shuffle_buffer_samples": 4,
        "log_interval_steps": 100,
        "val_interval_steps": 1,
        "checkpoint_dir": str(tmp_path / "grp"),
    }


def _grads_after_step(eu_loss_coef: float, tmp_path: Path) -> torch.Tensor:
    """相同种子与数据下执行一个训练步,返回末层 FC 梯度。"""
    torch.manual_seed(11)
    model = GRPModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = _base_config(tmp_path, eu_loss_coef)
    grp_train._train_step(
        model, optimizer, _fake_rows(4)[:2], torch.device("cpu"), config,
    )
    gradient = model.fc[-1].weight.grad
    assert gradient is not None
    return gradient.detach().clone()


def test_train_step_eu_coef_changes_gradients(tmp_path: Path) -> None:
    """eu_loss_coef>0 与 =0 的梯度不同:辅助项真实参与反向传播。"""
    with_eu = _grads_after_step(0.5, tmp_path)
    without_eu = _grads_after_step(0.0, tmp_path)
    assert not torch.allclose(with_eu, without_eu)


def test_train_step_eu_coef_zero_keeps_ce_gradients(tmp_path: Path) -> None:
    """eu_loss_coef=0 时与开关缺省(键缺失)的梯度完全一致(向后兼容)。"""
    explicit_zero = _grads_after_step(0.0, tmp_path)
    config = _base_config(tmp_path, 0.0)
    del config["eu_loss_coef"]
    torch.manual_seed(11)
    model = GRPModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    grp_train._train_step(
        model, optimizer, _fake_rows(4)[:2], torch.device("cpu"), config,
    )
    gradient = model.fc[-1].weight.grad
    assert gradient is not None
    torch.testing.assert_close(gradient.detach().clone(), explicit_zero, rtol=0, atol=0)


def test_train_grp_reloads_best_weights_before_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最终 val 差于中间最优时,best.pt 必须是中间最优权重且标签一致。"""
    config = _base_config(tmp_path, eu_loss_coef=0.0)
    rows = _fake_rows(6)
    monkeypatch.setattr(
        grp_train, "iter_grp_samples", lambda dataset, split: iter(rows),
    )

    state_at_best: dict[str, dict[str, torch.Tensor]] = {}
    calls = {"count": 0}

    def fake_validate(model, dataset, split, device, batch_size):
        calls["count"] += 1
        if calls["count"] == 1:
            # 第一次验证给出低 loss:随后 _maybe_save_best 落盘的正是当前权重,
            # 在此记录快照用于最终比对。
            state_at_best["state"] = copy.deepcopy({
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            })
            return 0.5, 0.01, float(len(rows))
        return 3.0, 0.5, float(len(rows))

    monkeypatch.setattr(grp_train, "evaluate_validation_loss", fake_validate)

    grp_train.train_grp(Path("unused-dataset"), config)

    payload = torch.load(
        tmp_path / "grp" / "best.pt", map_location="cpu", weights_only=False,
    )
    # loss 标签必须是中间最优,而非最终验证值。
    assert float(payload["validation_loss"]) == pytest.approx(0.5)
    assert (tmp_path / "grp" / "best_loss.json").is_file()
    # 落盘权重必须等于中间最优时刻的权重(而非训练最终权重)。
    saved = payload["model"]
    assert set(saved) == set(state_at_best["state"])
    for key, value in state_at_best["state"].items():
        torch.testing.assert_close(saved[key], value, rtol=0, atol=0)


def test_train_grp_final_best_keeps_final_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """最终 val 即最优时,落盘权重就是最终权重(无重载发生)。"""
    config = _base_config(tmp_path, eu_loss_coef=0.0)
    rows = _fake_rows(6)
    monkeypatch.setattr(
        grp_train, "iter_grp_samples", lambda dataset, split: iter(rows),
    )
    final_state: dict[str, dict[str, torch.Tensor]] = {}
    calls = {"count": 0}

    def fake_validate(model, dataset, split, device, batch_size):
        calls["count"] += 1
        # loss 逐步下降,最优恒出现在最后一次(最终)验证。
        loss = 3.0 - 0.1 * calls["count"]
        final_state["state"] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return loss, 0.1, float(len(rows))

    monkeypatch.setattr(grp_train, "evaluate_validation_loss", fake_validate)

    grp_train.train_grp(Path("unused-dataset"), config)

    payload = torch.load(
        tmp_path / "grp" / "best.pt", map_location="cpu", weights_only=False,
    )
    for key, value in final_state["state"].items():
        torch.testing.assert_close(payload["model"][key], value, rtol=0, atol=0)
