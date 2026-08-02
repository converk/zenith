from pathlib import Path

from riichi_ppo_v1.model.validation import run_random_coverage, write_coverage


output = Path(__file__).resolve().parents[1] / "action_roundtrip_random.json"
write_coverage(run_random_coverage(games=16, seed=20260802, max_steps=2500), output)
print(output)
