"""在线 bot 对局日志抓取与重放指标统计(riichi.dev)。

用途:针对 riichi.dev 上的某个 bot(bot_id,如 317):
  1. 通过 https://api.riichi.dev/api/v1/bots/<id>/games 翻页拉取该 bot 的
     全部对局元数据(game_id / played_at / seat / rank / score);
  2. 按日期范围本地过滤(实测 API 的 date_from/date_to 过滤返回异常,故不依赖);
  3. 并发下载每局日志
     https://logs.riichi.dev/mjai-logs/<yyyy>/<mm>/<dd>/<game_id>.jsonl.gz
     (mjai 协议格式,逐行 JSON);
  4. 重放日志,按与 riichi_ppo_v1/training/metrics.py 的 SemanticMetrics
     一致的口径逐小局统计:胡牌率 / 放铳率 / 副露率 / 平均和了 / 平均放铳 /
     流局率 / 破产率 / 平均巡目 等。

口径说明(与评测脚本对齐):
  - 分母一律为该 bot 参与的 player-kyoku(每个 start_kyoku 计一小局);
  - 胡牌:该 bot 座位为 hora.actor;放铳:hora.target == bot 座位 且
    actor != bot(自摸时 target==actor,需排除);
  - 平均和了 / 平均放铳:直接用 hora.deltas 中 bot 座位的点数(自摸为收入总和,
    荣和为荣和点数,与评测脚本 win_points/deal_in_points 同构),单位:点;
  - 流局:ryukyoku 事件计一局(含途中流局),荒牌流局单独统计
    (reason == "exhaustive_draw");
  - 破产:半庄结束(终局)时 bot 分数 < 0,与评测脚本 match_flying 口径一致;
  - 巡目:小局结束时"进行到第几巡",自摸 = 和了者摸牌次数,荣和 = 放铳者
    摸牌次数,流局 = 四家最大摸牌次数;
  - 半庄数:日志中出现 end_game 事件计一个完整半庄。

用法示例:
  # 抓取 bot 317 自 2026-08-31(含)起所有对局并统计
  python v18_online_bot_replay_metrics.py --bot-id 317 --date-from 2026-08-31
  # 复用已有缓存,不重新下载
  python v18_online_bot_replay_metrics.py --bot-id 317 --date-from 2026-08-31 --no-download
  # 全量统计(无日期过滤),并自动与官方 /stats 接口对照校验口径
  python v18_online_bot_replay_metrics.py --bot-id 317
  # 解析逻辑自检(不联网)
  python v18_online_bot_replay_metrics.py --self-test
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量(领域绑定,不随版本变化)
# ---------------------------------------------------------------------------

API_BASE = "https://api.riichi.dev/api/v1"
LOGS_BASE = "https://logs.riichi.dev/mjai-logs"
GAMES_PER_PAGE = 50  # 接口上限为 50
DEFAULT_WORKERS = 16
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_S = 60
USER_AGENT = "Mozilla/5.0 (bot-replay-stats; riichi.dev log analysis)"
MELD_TYPES = ("chi", "pon", "dai_min_kan", "kakan", "naki")
RIICHI_TYPES = ("reach", "riichi")  # 兼容两种 mjai 变体记法


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """API 返回的一局对局元数据。"""

    game_id: str
    played_at: str  # "YYYY-MM-DD HH:MM:SS"
    seat: int  # 该 bot 在当局的座位(0-3)
    rank: int | None = None
    score: int | None = None
    game_type: str = "ranked"
    is_disconnected: bool = False
    is_penalized: bool = False

    def log_url(self) -> str:
        """日志下载地址由 played_at 的日期与 game_id 拼出。"""
        day = self.played_at[:10].replace("-", "/")
        return f"{LOGS_BASE}/{day}/{self.game_id}.jsonl.gz"


@dataclass
class KyokuAccumulator:
    """一个小局内的重放中间状态。"""

    tsumo_counts: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    open_melds: int = 0  # 当前小局 bot 的副露数
    dahai_total: int = 0  # 当前小局四家弃牌总数(巡目参照)
    turns_kyoku: int | None = None  # 当前小局"进行到第几巡"
    finalized: bool = False  # 小局是否已出现终结事件(hora/ryukyoku)


@dataclass
class ReplayResult:
    """单局重放结果:小局级统计与半庄级统计。"""

    kyoku: int = 0
    wins: int = 0
    tsumo_wins: int = 0
    ron_wins: int = 0
    deal_ins: int = 0
    tsumo_losses: int = 0  # 其他家自摸时 bot 失点(被自摸)
    ryukyoku: int = 0
    exhaustive_draws: int = 0
    riichis: int = 0
    furo_kyoku: int = 0  # 做了副露的小局数
    open_melds: int = 0  # 副露数合计
    win_points: list[int] = field(default_factory=list)
    deal_in_points: list[int] = field(default_factory=list)
    turns_all: list[int] = field(default_factory=list)  # 每小局巡目
    win_turns: list[int] = field(default_factory=list)  # 和了小局的巡目
    final_score: int | None = None  # 半庄终局时 bot 分数
    rank: int | None = None  # 半庄终局名次(0-3 → 1-4)
    full_hanchan: bool = False  # 是否看到 end_game(完整重放)
    error: str | None = None  # 重放异常信息


# ---------------------------------------------------------------------------
# HTTP 与缓存
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = DOWNLOAD_TIMEOUT_S) -> bytes:
    last_error: Exception | None = None
    for _attempt in range(DOWNLOAD_RETRIES):
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 * (_attempt + 1))
    raise RuntimeError(f"GET {url} failed after {DOWNLOAD_RETRIES} retries: {last_error}")


def _http_get_json(url: str) -> dict:
    return json.loads(_http_get(url).decode("utf-8"))


def fetch_bot_games(bot_id: int) -> list[GameRecord]:
    """翻页拉取该 bot 的全部对局元数据(游标分页,按 played_at 倒序)。"""
    base = f"{API_BASE}/bots/{bot_id}/games?limit={GAMES_PER_PAGE}"
    url: str | None = base
    records: list[GameRecord] = []
    seen: set[str] = set()
    while url is not None:
        payload = _http_get_json(url)
        data = payload["data"]
        for row in data.get("games", []):
            if row["game_id"] not in seen:
                seen.add(row["game_id"])
                records.append(
                    GameRecord(
                        game_id=row["game_id"],
                        played_at=row["played_at"],
                        seat=int(row["seat"]),
                        rank=row.get("rank"),
                        score=row.get("score"),
                        game_type=row.get("game_type", "ranked"),
                        is_disconnected=bool(row.get("is_disconnected", False)),
                        is_penalized=bool(row.get("is_penalized", False)),
                    )
                )
        if data.get("has_more") and data.get("next_cursor"):
            url = base + "&cursor=" + urllib.parse.quote(data["next_cursor"], safe="")
        else:
            url = None
        time.sleep(0.05)  # 避免打爆接口
    return records


def fetch_bot_profile(bot_id: int) -> dict:
    return _http_get_json(f"{API_BASE}/bots/{bot_id}")["data"]


def fetch_bot_stats(bot_id: int) -> dict:
    """官方全量统计接口(仅用于口径自检,不能按日期过滤)。"""
    return _http_get_json(f"{API_BASE}/bots/{bot_id}/stats")["data"]


def _cache_path(cache_dir: Path, record: GameRecord) -> Path:
    day = record.played_at[:10]
    return cache_dir / day / f"{record.game_id}.jsonl.gz"


def fetch_log(
    record: GameRecord, cache_dir: Path, *, allow_download: bool
) -> tuple[GameRecord, Path | None, str | None]:
    """下载(或使用缓存)一局日志。返回 (record, 本地路径, 错误信息)。"""
    path = _cache_path(cache_dir, record)
    if path.exists() and path.stat().st_size > 0:
        return record, path, None
    if not allow_download:
        return record, None, "cache miss (no-download)"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    for _attempt in range(DOWNLOAD_RETRIES):
        try:
            data = _http_get(record.log_url())
            with open(tmp, "wb") as file:
                file.write(data)
            os.replace(tmp, path)
            return record, path, None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            try:
                os.remove(tmp)
            except OSError:
                pass
            last_error = f"{type(exc).__name__}: {exc}"
    return record, None, last_error


# ---------------------------------------------------------------------------
# 日志重放
# ---------------------------------------------------------------------------


def replay_log(path: Path, seat: int) -> ReplayResult:
    """重放一局 mjai 日志,统计该 bot(座位 seat)的小局级与半庄级指标。

    结算口径:
    - 小局级(副露数 / 巡目)统一在 end_kyoku 结算一次,避免同一小局出现
      多个终结事件(如多荣和时 2 个 hora)时被重复计数;
    - 终局分数:每小局 start_kyoku 的 scores(已扣除立直棒等)为权威起点,
      加本小局 hora/ryukyoku 的 deltas 累计,多小局叠加得到半庄终局分,
      这与官方口径一致(纯 deltas 累计会漏掉立直棒的扣款)。
    """
    result = ReplayResult()
    acc = KyokuAccumulator()
    kyoku_start_scores: list[int] | None = None  # 当前小局开始时的权威分数
    kyoku_delta_acc: list[int] = [0, 0, 0, 0]  # 当前小局 deltas 累计
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (TypeError, ValueError) as exc:
                if result.error is None:
                    result.error = f"bad json line: {exc}"
                continue
            event_type = event.get("type")

            if event_type == "start_kyoku":
                result.kyoku += 1
                acc = KyokuAccumulator()
                kyoku_start_scores = [int(v) for v in event.get("scores", [0] * 4)]
                kyoku_delta_acc = [0, 0, 0, 0]

            elif event_type == "tsumo":
                actor = int(event["actor"])
                acc.tsumo_counts[actor] += 1

            elif event_type == "dahai":
                acc.dahai_total += 1

            elif event_type in MELD_TYPES:
                if int(event.get("actor", -1)) == seat:
                    acc.open_melds += 1

            elif event_type in RIICHI_TYPES:
                if int(event.get("actor", -1)) == seat:
                    result.riichis += 1

            elif event_type == "hora":
                deltas = [int(v) for v in event.get("deltas", [0] * 4)]
                if len(deltas) == 4:
                    kyoku_delta_acc = [s + d for s, d in zip(kyoku_delta_acc, deltas)]
                actor = int(event.get("actor", -1))
                target = event.get("target")
                is_tsumo = bool(event.get("tsumo"))
                # 巡目:自摸 = 和了者摸牌次数,荣和 = 放铳者摸牌次数
                turns = (
                    acc.tsumo_counts[actor] if is_tsumo
                    else (acc.tsumo_counts[int(target)] if target is not None else max(acc.tsumo_counts))
                )
                if acc.turns_kyoku is None:
                    acc.turns_kyoku = turns
                if actor == seat:
                    result.wins += 1
                    result.tsumo_wins += int(is_tsumo)
                    result.ron_wins += int(not is_tsumo)
                    result.win_points.append(deltas[seat])
                    result.win_turns.append(turns)
                elif target == seat:
                    # bot 放铳(荣和且 target == bot 座位)
                    result.deal_ins += 1
                    result.deal_in_points.append(
                        -deltas[seat] if deltas[seat] < 0 else deltas[seat]
                    )
                if actor != seat and is_tsumo:
                    # 其他家自摸,bot 必定失点(被自摸)
                    result.tsumo_losses += 1
                acc.finalized = True

            elif event_type == "ryukyoku":
                deltas = [int(v) for v in event.get("deltas", [0] * 4)]
                if len(deltas) == 4:
                    kyoku_delta_acc = [s + d for s, d in zip(kyoku_delta_acc, deltas)]
                result.ryukyoku += 1
                if event.get("reason") == "exhaustive_draw":
                    result.exhaustive_draws += 1
                if acc.turns_kyoku is None:
                    acc.turns_kyoku = max(acc.tsumo_counts)
                acc.finalized = True

            elif event_type == "end_kyoku":
                # 小局级结算,每个小局只计一次(多终结事件时靠 finalized 防漏防重)
                if acc.turns_kyoku is None:
                    acc.turns_kyoku = max(acc.tsumo_counts)
                result.furo_kyoku += int(acc.open_melds > 0)
                result.open_melds += acc.open_melds
                result.turns_all.append(acc.turns_kyoku)

            elif event_type == "end_game":
                result.full_hanchan = True
                if kyoku_start_scores is not None:
                    final = [
                        s + d for s, d in zip(kyoku_start_scores, kyoku_delta_acc)
                    ]
                    result.final_score = final[seat]
                    ranking = sorted(range(4), key=lambda s: (-final[s], s))
                    result.rank = ranking.index(seat) + 1

    if not result.full_hanchan and result.final_score is not None:
        if result.error is None:
            result.error = "incomplete log (no end_game)"
    return result


# ---------------------------------------------------------------------------
# 聚合统计
# ---------------------------------------------------------------------------


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def aggregate(results: list[ReplayResult]) -> dict:
    """把多局重放结果汇总为指标字典(口径与评测脚本一致)。"""
    kyoku_total = sum(r.kyoku for r in results)
    wins = sum(r.wins for r in results)
    deal_ins = sum(r.deal_ins for r in results)
    ryukyoku = sum(r.ryukyoku for r in results)
    hanchan = sum(1 for r in results if r.full_hanchan)
    win_points = [p for r in results for p in r.win_points]
    deal_in_points = [p for r in results for p in r.deal_in_points]
    turns_all = [t for r in results for t in r.turns_all]
    win_turns = [t for r in results for t in r.win_turns]
    ranks = [r.rank for r in results if r.rank is not None]
    final_scores = [r.final_score for r in results if r.final_score is not None]
    kyoku_lengths = [r.kyoku for r in results if r.full_hanchan]

    metrics = {
        "kyoku_count": kyoku_total,
        "win_count": wins,
        "win_rate": _rate(wins, kyoku_total),
        "tsumo_rate": _rate(sum(r.tsumo_wins for r in results), kyoku_total),
        "ron_rate": _rate(sum(r.ron_wins for r in results), kyoku_total),
        "deal_in_count": deal_ins,
        "deal_in_rate": _rate(deal_ins, kyoku_total),
        "tsumo_loss_rate": _rate(sum(r.tsumo_losses for r in results), kyoku_total),
        "furo_kyoku_count": sum(r.furo_kyoku for r in results),
        "furo_rate": _rate(sum(r.furo_kyoku for r in results), kyoku_total),
        "open_melds_total": sum(r.open_melds for r in results),
        "open_melds_mean": _rate(sum(r.open_melds for r in results), kyoku_total),
        "win_points_sum": sum(win_points),
        "win_points_mean": _mean(win_points),  # 平均和了(点)
        "deal_in_points_sum": sum(abs(p) for p in deal_in_points),
        "deal_in_points_mean": _mean([abs(p) for p in deal_in_points]),  # 平均放铳(点)
        "ryukyoku_count": ryukyoku,
        "ryukyoku_rate": _rate(ryukyoku, kyoku_total),
        "exhaustive_draw_count": sum(r.exhaustive_draws for r in results),
        "exhaustive_draw_rate": _rate(sum(r.exhaustive_draws for r in results), kyoku_total),
        "riichi_count": sum(r.riichis for r in results),
        "riichi_rate": _rate(sum(r.riichis for r in results), kyoku_total),
        "tobi_count": sum(1 for s in final_scores if s < 0),
        "tobi_rate": _rate(sum(1 for s in final_scores if s < 0), hanchan),  # 破产率
        "hanchan_count": hanchan,
        "avg_turns": _mean(turns_all),  # 平均巡目(全部小局)
        "avg_win_turns": _mean(win_turns),  # 平均和了巡目
        "avg_kyoku_per_hanchan": _mean(kyoku_lengths),
        "avg_final_score": _mean(final_scores),
        "avg_rank": _mean(ranks),
        "rank_distribution": dict(Counter(ranks)),
    }
    return metrics


# ---------------------------------------------------------------------------
# 自检口(不依赖网络)
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """用一段构造日志验证重放器关键口径,失败返回 False。"""
    log_rows = [
        {"type": "start_game"},
        {"type": "start_kyoku", "scores": [25000, 25000, 25000, 25000],
         "tehais": [[], [], [], []], "oya": 0},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "dahai", "actor": 0, "pai": "5m"},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "dahai", "actor": 1, "pai": "6m"},
        {"type": "tsumo", "actor": 2, "pai": "3m"},
        {"type": "dahai", "actor": 2, "pai": "7m"},
        {"type": "tsumo", "actor": 3, "pai": "4m"},
        # 第 1 巡末,bot(seat 1)第 1 次摸牌后流局(仅验证计数逻辑)
        {"type": "ryukyoku", "reason": "exhaustive_draw",
         "deltas": [1000, -1000, 0, 0], "tenpai": [0], "noten": [1, 2, 3]},
        {"type": "end_kyoku"},
        {"type": "start_kyoku", "scores": [26000, 24000, 25000, 25000],
         "tehais": [[], [], [], []], "oya": 1},
        {"type": "tsumo", "actor": 1, "pai": "5p"},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9p"},
        {"type": "tsumo", "actor": 2, "pai": "6p"},
        {"type": "dahai", "actor": 2, "pai": "9m"},
        {"type": "tsumo", "actor": 3, "pai": "7p"},
        {"type": "dahai", "actor": 3, "pai": "8m"},
        {"type": "tsumo", "actor": 0, "pai": "8p"},
        {"type": "dahai", "actor": 0, "pai": "1s"},
        {"type": "tsumo", "actor": 1, "pai": "5m"},
        {"type": "dahai", "actor": 1, "pai": "1p"},
        {"type": "tsumo", "actor": 2, "pai": "6m"},
        {"type": "dahai", "actor": 2, "pai": "2s"},
        {"type": "tsumo", "actor": 3, "pai": "7m"},
        {"type": "dahai", "actor": 3, "pai": "3s"},
        {"type": "tsumo", "actor": 0, "pai": "8m"},
        {"type": "dahai", "actor": 0, "pai": "4s"},
        {"type": "hora", "actor": 1, "target": 0, "tsumo": False,
         "deltas": [0, 9700, 0, 0], "ura_markers": []},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    payload = "\n".join(json.dumps(row) for row in log_rows).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(payload)
    # 临时文件写盘后重放,结束后清理
    tmp_path = Path("/tmp/_replay_selftest.jsonl.gz")
    tmp_path.write_bytes(buf.getvalue())
    result = replay_log(tmp_path, seat=1)
    tmp_path.unlink(missing_ok=True)
    checks = [
        result.kyoku == 2,
        result.wins == 1 and result.ron_wins == 1 and result.tsumo_wins == 0,
        result.deal_ins == 0,
        result.riichis == 1,
        result.ryukyoku == 1 and result.exhaustive_draws == 1,
        result.turns_all == [1, 2] and result.win_turns == [2],
        result.full_hanchan,
        result.final_score == 24000 + 9700 and result.rank == 1,
    ]
    for ok, name in zip(checks, [
        "kyoku count", "win/ron/tsumo", "deal-in", "riichi",
        "ryukyoku", "turns", "end_game", "final score & rank",
    ]):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _print_table(metrics: dict, bot_profile: dict, date_from: str, date_to: str) -> None:
    print("=" * 64)
    print(f"bot       : {bot_profile.get('name')} (id={bot_profile.get('id')}, "
          f"rating={bot_profile.get('rating')})")
    print(f"日期范围  : {date_from} 00:00:00 ~ {date_to} 23:59:59")
    print(f"半庄数    : {metrics['hanchan_count']}   小局数: {metrics['kyoku_count']}")
    print("-" * 64)
    rows = [
        ("胡牌率", f"{metrics['win_rate']:.2%}  ({metrics['win_count']}/{metrics['kyoku_count']})"),
        ("放铳率", f"{metrics['deal_in_rate']:.2%}  ({metrics['deal_in_count']}/{metrics['kyoku_count']})"),
        ("副露率", f"{metrics['furo_rate']:.2%}  ({metrics['furo_kyoku_count']}/{metrics['kyoku_count']})"),
        ("平均每小局副露数", f"{metrics['open_melds_mean']:.3f}"),
        ("平均和了", f"{metrics['win_points_mean']:,.0f} 点  ({metrics['win_count']} 次)"),
        ("平均放铳", f"{metrics['deal_in_points_mean']:,.0f} 点  ({metrics['deal_in_count']} 次)"),
        ("流局率", f"{metrics['ryukyoku_rate']:.2%}  ({metrics['ryukyoku_count']}/{metrics['kyoku_count']})"),
        ("(其中荒牌流局率)", f"{metrics['exhaustive_draw_rate']:.2%}"),
        ("破产率", f"{metrics['tobi_rate']:.2%}  ({metrics['tobi_count']}/{metrics['hanchan_count']})"),
        ("平均巡目", f"{metrics['avg_turns']:.2f}"),
        ("平均和了巡目", f"{metrics['avg_win_turns']:.2f}"),
        ("自摸率", f"{metrics['tsumo_rate']:.2%}"),
        ("荣和率", f"{metrics['ron_rate']:.2%}"),
        ("被自摸率", f"{metrics['tsumo_loss_rate']:.2%}"),
        ("立直率", f"{metrics['riichi_rate']:.2%}"),
        ("平均顺位", f"{metrics['avg_rank']:.3f}"),
        ("平均终局分", f"{metrics['avg_final_score']:,.0f}"),
        ("每半庄平均小局数", f"{metrics['avg_kyoku_per_hanchan']:.2f}"),
    ]
    for name, value in rows:
        print(f"  {name:<18s}: {value}")
    print("=" * 64)


def _check_against_api(result_all: list[ReplayResult], stats: dict) -> None:
    """全量口径自检:与官方 /stats 接口逐项对照(仅全量无过滤时调用)。"""
    kyoku_total = sum(r.kyoku for r in result_all)
    pairs = [
        ("kyoku", "total_kyoku"),
        ("wins", "agari_count"),
        ("tsumo_wins", "tsumo_count"),
        ("deal_ins", "houju_count"),
        ("riichis", "riichi_count"),
        ("furo_kyoku", "furo_kyoku_count"),
        ("ryukyoku", "ryukyoku_count"),
    ]
    print("官方 /stats 口径自检(全量):")
    ok = True
    for local_key, api_key in pairs:
        local = sum(getattr(r, local_key) for r in result_all)
        api = int(stats[api_key])
        # 抓取与统计接口之间存在进行中对局,允许少量增量差
        tolerance = 8 if local_key == "kyoku" else 2
        match = abs(local - api) <= tolerance
        ok &= match
        print(f"  [{'PASS' if match else 'DIFF'}] {local_key}: {local} vs api {api}")
    local_tobi = sum(1 for r in result_all if r.final_score is not None and r.final_score < 0)
    print(f"  tobi_count(直接计算): {local_tobi} vs api {stats['tobi_count']}")
    win_pts = sum(p for r in result_all for p in r.win_points)
    print(f"  win_points_sum: {win_pts} vs api {stats['agari_points_sum']}")
    houju_pts = sum(abs(p) for r in result_all for p in r.deal_in_points)
    print(f"  deal_in_points_sum: {houju_pts} vs api {stats['houju_points_sum']}")
    print("  自检结论:", "OK" if ok else "存在差异,请检查口径")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-id", type=int, default=317, help="riichi.dev bot id")
    parser.add_argument("--date-from", default=None,
                        help="起始日期(含),格式 YYYY-MM-DD")
    parser.add_argument("--date-to", default=None,
                        help="截止日期(含),格式 YYYY-MM-DD;缺省为今天")
    parser.add_argument("--cache-dir", default=None,
                        help="日志缓存目录;缺省 ~/.cache/riichidev-bot-logs/<bot_id>")
    parser.add_argument("--no-download", action="store_true",
                        help="只使用已有缓存,不下载新日志")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", default=None,
                        help="JSON 输出路径;缺省 audit/reports/v18/eval/ 下按范围命名")
    parser.add_argument("--self-test", action="store_true", help="重放器自检后退出")
    args = parser.parse_args()

    if args.self_test:
        ok = _self_test()
        print("self-test:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser()
    else:
        cache_dir = Path.home() / ".cache" / "riichidev-bot-logs" / str(args.bot_id)

    print(f"拉取 bot {args.bot_id} 对局元数据 ...", flush=True)
    profile = fetch_bot_profile(args.bot_id)
    all_games = fetch_bot_games(args.bot_id)
    now = time.strftime("%Y-%m-%d")
    date_from = args.date_from or all_games[-1].played_at[:10]
    date_to = args.date_to or now
    lower = f"{date_from} 00:00:00"
    upper = f"{date_to} 23:59:59"
    in_range = [
        g for g in all_games
        if lower <= g.played_at <= upper
    ]
    print(f"元数据共 {len(all_games)} 局,日期范围 "
          f"{date_from}~{date_to} 命中 {len(in_range)} 局", flush=True)

    print(f"下载/读取日志(缓存 {cache_dir},并发 {args.workers}) ...", flush=True)
    start = time.perf_counter()
    fetched = [None] * len(in_range)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_log, record, cache_dir, allow_download=not args.no_download)
            for record in in_range
        ]
        for future, record in zip(futures, in_range):
            _gid, path, error = future.result()
            fetched[in_range.index(record)] = (record, path, error)
    elapsed = time.perf_counter() - start
    ok_paths = [(r, p) for r, p, e in fetched if p is not None and e is None]
    failed = [(r, e) for r, p, e in fetched if e is not None]
    print(f"日志就绪 {len(ok_paths)}/{len(in_range)} 局,失败 {len(failed)} "
          f"(耗时 {elapsed:.1f}s)", flush=True)

    results: list[ReplayResult] = []
    failures: list[dict] = []
    for record, path in ok_paths:
        try:
            result = replay_log(path, record.seat)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - 单局失败不中断整体统计
            failures.append({"game_id": record.game_id, "played_at": record.played_at,
                             "error": f"{type(exc).__name__}: {exc}"})
    for record, err in failed:
        failures.append({"game_id": record.game_id, "played_at": record.played_at,
                         "error": err})
    print(f"重放成功 {len(results)} 局,异常 {len(failures)} 局", flush=True)

    metrics = aggregate(results)
    print()
    _print_table(metrics, profile, date_from, date_to)
    if failures:
        print(f"异常对局 {len(failures)} 局(仅记录,未计指标):")
        for item in failures[:10]:
            print(f"  {item['played_at']} {item['game_id']} -- {item['error']}")
        if len(failures) > 10:
            print(f"  ... 其余 {len(failures) - 10} 局见 JSON 输出")

    report = {
        "schema_version": 1,
        "bot": profile,
        "date_from_inclusive": date_from,
        "date_to_inclusive": date_to,
        "games_meta_total": len(all_games),
        "games_in_range": len(in_range),
        "games_replayed_ok": len(results),
        "game_failures": failures,
        "metrics": metrics,
    }

    # 全量口径自检:未带日期过滤时与官方 /stats 对照
    if args.date_from is None and args.date_to is None:
        stats = fetch_bot_stats(args.bot_id)
        _check_against_api(results, stats)
        report["api_stats_check"] = stats

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "eval" /
        f"bot{args.bot_id}_replay_metrics_{date_from}_to_{date_to}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output)
    print(f"\nJSON 报告: {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
