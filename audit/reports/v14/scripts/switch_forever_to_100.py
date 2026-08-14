"""Watch the two ranked bots and switch each from --forever to --games 100
at the next clean game boundary (after the current game ends and before the
next game starts)."""

from __future__ import annotations

import json
import subprocess
import time


SESSIONS = {
    1: {
        "log": "logs/v14/ranked_sft_forever.jsonl",
        "new_log": "logs/v14/ranked_sft_100_v2.jsonl",
        "marker": "S1_FOREVER_EXITED",
        "command": (
            "cd /mnt/disk1/hubowen/zenith && CUDA_DEVICE=0 riichi-lab-bot "
            "ranked --games 100 --device cuda:0 --dtype fp32 "
            "--checkpoint checkpoints/train_riichi_v13/sft/best_heuristic.pt "
            "--jsonl-log logs/v14/ranked_sft_100_v2.jsonl"
        ),
    },
    2: {
        "log": "logs/v14/ranked_u510_b_forever.jsonl",
        "new_log": "logs/v14/ranked_u510_b_100_v2.jsonl",
        "marker": "S2_FOREVER_EXITED",
        "command": (
            "cd /mnt/disk1/hubowen/zenith && CUDA_DEVICE=0 riichi-lab-bot "
            "ranked --games 100 --device cuda:0 --dtype fp32 "
            "--checkpoint checkpoints/train_riichi_v14/"
            "bot_checkpoint_00510_sft_wrapper.pt "
            "--jsonl-log logs/v14/ranked_u510_b_100_v2.jsonl"
        ),
    },
}


def events(log: str) -> list[dict]:
    try:
        with open(log, encoding="utf-8") as handle:
            return [
                json.loads(line)
                for line in handle
                if line.strip()
            ]
    except FileNotFoundError:
        return []


def last_game_order(log: str) -> tuple[float, float]:
    ended = 0.0
    started = 0.0
    for event in events(log):
        stamp = float(event["timestamp"])
        name = event.get("event")
        if name == "game_ended":
            ended = stamp
        elif name == "game_started":
            started = stamp
    return ended, started


def pane_has(session: int, text: str) -> bool:
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", f"{session}:0", "-p"],
        capture_output=True,
        text=True,
    )
    return text in result.stdout


def send(session: int, keys: str, *, enter: bool = True) -> None:
    command = ["tmux", "send-keys", "-t", f"{session}:0", keys]
    if enter:
        command.append("Enter")
    subprocess.run(command, check=True)


def main() -> None:
    baseline: dict[int, int] = {
        session: sum(1 for event in events(info["log"]) if event.get("event") == "game_ended")
        for session, info in SESSIONS.items()
    }
    print(f"baseline game_ended: {baseline}", flush=True)
    switched: set[int] = set()
    while len(switched) < len(SESSIONS):
        for session, info in SESSIONS.items():
            if session in switched:
                continue
            current = sum(
                1 for event in events(info["log"])
                if event.get("event") == "game_ended"
            )
            if current <= baseline[session]:
                continue
            ended, started = last_game_order(info["log"])
            if started > ended:
                # A new game already began; switch after that game ends.
                baseline[session] = current
                print(
                    f"session {session}: game {current} ended but next game "
                    f"already started; waiting for it to end",
                    flush=True,
                )
                continue
            print(
                f"session {session}: game {current} ended at boundary; "
                f"interrupting --forever",
                flush=True,
            )
            send(session, "C-c")
            deadline = time.time() + 30
            while time.time() < deadline:
                if pane_has(session, info["marker"]):
                    break
                time.sleep(1)
            else:
                print(
                    f"session {session}: marker not seen, checking prompt",
                    flush=True,
                )
            time.sleep(2)
            print(f"session {session}: launching --games 100", flush=True)
            send(session, info["command"])
            switched.add(session)
            baseline[session] = 0
        time.sleep(3)
    print("BOTH_SWITCHED", flush=True)


if __name__ == "__main__":
    main()
