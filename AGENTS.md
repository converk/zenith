# Project Facts

- Use `CUDA_DEVICE=0,3` and `learner_gpus=2` by default for performance and
  training tests; device `3` corresponds to physical GPU 4. Use
  `CUDA_DEVICE=0` only when a single-GPU run is explicitly requested. The
  training entry point maps `CUDA_DEVICE` to CUDA's standard
  `CUDA_VISIBLE_DEVICES` before starting PyTorch or Ray.
- `CUDA_DEVICE=0` (also referred to as `CUDA=0`) maps to physical GPU 0.
- `CUDA_DEVICE=1` (also referred to as `CUDA=1`) maps to physical GPU 1.
- `CUDA_DEVICE=2` (also referred to as `CUDA=2`) maps to physical GPU 3.
- `CUDA_DEVICE=3` (also referred to as `CUDA=3`) maps to physical GPU 4.
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
- When running tests, print elapsed-time monitoring and all relevant
  performance metrics by default. Run three iterations by default; treat the
  first as a potential warm-up and report performance statistics for the
  subsequent iterations separately.

# Project Structure and File Organization

- Keep the project structure clean when modifying code. Do not accumulate
  unrelated responsibilities in an existing file merely because it is already
  nearby.
- When a component has an independent responsibility, create a new file and
  place it in the appropriate existing directory instead of forcing the code
  into an unrelated module.
- Before adding a new file, check the current package layout and naming
  conventions; keep related implementation, tests, documentation, and
  configuration in their corresponding directories.
