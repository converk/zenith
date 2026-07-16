# Project Facts

- Use `CUDA_DEVICE=0` by default for GPU commands. The training entry point
  maps it to CUDA's standard `CUDA_VISIBLE_DEVICES` before starting PyTorch or
  Ray.
- Use the Conda environment named `Mahjong-AI` for Python commands and training.
- `RiichiEnv` is this project's training environment.
- `exp/` contains the existing training framework used as the reference when
  designing or comparing this project's training implementation.
- `riichi_ppo_v1/` is the primary training-code framework for this project.
- Compatibility with `evaluations/` is not required; that component will be
  rewritten.
- Default performance and training tests must explicitly use
  `target_kl=0.0`, `update_epochs=4`, and `kyokus_per_worker=1`. This test
  baseline is independent of the long-running training default, whose
  `kyokus_per_worker` remains `16`.
