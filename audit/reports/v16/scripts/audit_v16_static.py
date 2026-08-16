"""V16 静态契约审计:协议常量、文档与 Rust 编码语义的一致性。

本脚本只读仓库文件与已安装扩展,不修改任何训练产物。任何不一致都以非零退出码
报告,供上层审计流程汇总。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)
    raise SystemExit(1)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        fail(f"{name}: {detail}")
    print(f"[PASS] {name}", flush=True)


def main() -> None:
    from riichi_ppo_v1.model import encoding_protocol as protocol
    from riichi_ppo_v1.model.schema import TOKEN_SCHEMA_VERSION
    from riichi_ppo_v1.sft.contract import V16_ACTOR_INPUT_CONTRACT_SHA256

    # 契约内容哈希必须与数据集 manifest 使用的常量一致。
    contract = ROOT / "specs/003-v16-model-rework/contracts/actor-input-v16.md"
    actual_hash = hashlib.sha256(contract.read_bytes()).hexdigest()
    check(
        "actor-input-v16.md 内容哈希与冻结常量一致",
        actual_hash == V16_ACTOR_INPUT_CONTRACT_SHA256,
        f"actual={actual_hash} frozen={V16_ACTOR_INPUT_CONTRACT_SHA256}",
    )

    check(
        "协议版本唯一且格式标识由常量派生",
        protocol.ENCODING_PROTOCOL_VERSION == 16
        and protocol.ENCODED_FORMAT == "riichi-sft-encoded-v16",
    )
    check(
        "token schema 版本引用协议常量",
        TOKEN_SCHEMA_VERSION == protocol.ENCODING_PROTOCOL_VERSION,
        f"got {TOKEN_SCHEMA_VERSION}",
    )

    expected_cardinalities = {
        "O0": 7, "O1": 11, "O2": 7, "O3": 14, "O4": 4, "O5": 6,
        "O6": 4, "O7": 2, "O8": 3, "O9": 6,
        "D0": 3, "D1": 3, "D2": 3, "D3": 3, "D4": 3, "D5": 3,
        "D6": 5, "D7": 5, "D8": 5, "D9": 6,
    }
    check(
        "20 个 slot 基数与契约一致",
        protocol.SLOT_CARDINALITIES == expected_cardinalities,
        f"got {protocol.SLOT_CARDINALITIES}",
    )
    check(
        "标签表基数与 SLOT_CARDINALITIES 一致",
        all(
            len(protocol.OFFENSE_SLOT_LABELS[slot]) == protocol.SLOT_CARDINALITIES[slot]
            for slot in protocol.OFFENSE_SLOT_ORDER
        )
        and all(
            len(protocol.DEFENSE_SLOT_LABELS[slot]) == protocol.SLOT_CARDINALITIES[slot]
            for slot in protocol.DEFENSE_SLOT_ORDER
        ),
    )
    check(
        "O3-O6 N/A 位于编码 0,D0-D5/O8/D9 N/A 位于末位",
        all(protocol.OFFENSE_SLOT_LABELS[slot][0] == "N/A" for slot in ("O3", "O4", "O5", "O6"))
        and all(protocol.DEFENSE_SLOT_LABELS[slot][-1] == "N/A" for slot in ("D0", "D1", "D2", "D3", "D4", "D5"))
        and protocol.OFFENSE_SLOT_LABELS["O8"][-1] == "N/A"
        and protocol.DEFENSE_SLOT_LABELS["D9"][-1] == "N/A",
    )
    check(
        "bucket 边界符合契约例示值",
        [protocol.bucket_o1(v) for v in (0, 3, 7, 9, 10, 12)] == [0, 3, 7, 9, 10, 10]
        and [protocol.bucket_o2(v) for v in (0, 1, 4, 5, 9, 12, 13, 17, 21, 52)]
        == [0, 1, 1, 2, 3, 3, 4, 5, 6, 6]
        and [protocol.bucket_o3(v) for v in (None, 1, 2, 13, 14)] == [0, 1, 2, 13, 13]
        and [protocol.bucket_o5(v) for v in (None, 1, 4, 5, 13)] == [0, 1, 4, 5, 5]
        and [protocol.bucket_o9(v) for v in (0, 4, 5, 6)] == [0, 4, 5, 5]
        and [protocol.bucket_d6(v) for v in (0, 3, 4, 7)] == [0, 3, 4, 4]
        and [protocol.bucket_d9(v) for v in (None, 0, 4, 5)] == [5, 0, 4, 4],
    )

    # 实现侧文档必须引用同一权威常量和格式。
    protocol_doc = (ROOT / "riichi_ppo_v1/docs/v16_input_protocol.md").read_text(encoding="utf-8")
    check(
        "v16_input_protocol.md 引用单一协议常量",
        "encoding_protocol.py" in protocol_doc and "QUERY_ROW_WIDTH=15" in protocol_doc,
    )

    # 当前编码数据集的 manifest 必须是 v16 单版本契约。
    dataset_manifest = ROOT / "datasets/tenhou_sft_2024_2025_encoded_40pct_v16/manifest.json"
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    check(
        "当前 V16 数据集 manifest 协议版本与哈希一致",
        manifest.get("format") == protocol.ENCODED_FORMAT
        and manifest.get("encoding_protocol_version") == 16
        and manifest.get("encoding_contract_sha256") == V16_ACTOR_INPUT_CONTRACT_SHA256,
        f"got format={manifest.get('format')!r}",
    )

    # Rust 扩展与 Python 边界必须保持同一 runtime 契约。
    import riichi
    import riichienv

    check(
        "安装的 Rust 扩展 runtime 契约一致",
        getattr(riichi, "ANALYSIS_VERSION", None) == 4
        and getattr(riichienv, "REPLAY_SEMANTICS_VERSION", None) == 1,
        f"riichi={getattr(riichi, 'ANALYSIS_VERSION', None)!r} "
        f"riichienv={getattr(riichienv, 'REPLAY_SEMANTICS_VERSION', None)!r}",
    )

    print("static audit: all checks passed", flush=True)


if __name__ == "__main__":
    main()
