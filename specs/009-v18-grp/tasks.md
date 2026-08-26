# Tasks: V18 GRP 输入扩展与重新训练

**Input**: `specs/009-v18-grp/`(spec.md、plan.md)
**Tests**: 特性任务要求;故事测试任务先于实现任务。
**Organization**: 按依赖顺序执行;每项完成并验证后才标记 `[X]`。

## Phase 1: 契约与模型

- [ ] T001 更新 `riichi_ppo_v1/model/grp.py`:GRP_INPUT_SIZE=21(带布局注释)、
  GRP_HIDDEN=96、GRPModel(input_size/hidden_size/num_layers/num_classes 可配置),
  forward 校验 `features.shape[-1] == input_size`。
- [ ] T002 更新 `riichi_ppo_v1/training/grp/prepare.py`:新增
  `game_type_from_content`/`game_type_from_mode`/`result_increment`/`feature_row`,
  重写 `features_from_boundaries(boundaries, game_type)` 按边界链推进累计计数;
  dataset.json format=`riichi-grp-v18`、input_size=21、game_type 分布。
- [ ] T003 更新 `riichi_ppo_v1/training/grp/train.py`:model_config 快照与
  checkpoint 使用 GRP_INPUT_SIZE/GRP_HIDDEN/GRP_LAYERS 常量,新增
  feature_layout 记录;文档字符串同步 21 维/96。

## Phase 2: 运行时装配

- [ ] T004 更新 `tiichi_ppo_v1/training/worker.py`:GrpRollout 构造带 game_type,
  每环境维护累计计数(wins/dealins/tenpai),边界行改由 `feature_row` 生成;
  GRP 模型按 checkpoint model_config 构造;注释与错误信息同步新规模。

## Phase 3: 配置与脚本

- [ ] T005 新增 `riichi_ppo_v1/configs/v18_grp.yaml`(自包含超参与
  checkpoint_dir)。
- [ ] T006 新增 `audit/reports/v18/scripts/run_v18_grp_prepare_and_train.sh`
  (prepare → train,tee 到 logs/v18/,--skip-prepare)。

## Phase 4: 测试与文档

- [ ] T007 [P] 更新 `tests/unit/test_grp_mortal.py`:21 维特征行断言、累计计数
  推进、局风映射、参数预算 110K–150K、离线/在线一致性、dataset.json v18 格式。
- [ ] T008 [P] 更新 `tests/unit/test_v17_reward.py`:GrpRollout 新签名。
- [ ] T009 运行 `pytest -q riichi_ppo_v1/tests`,全绿并记录结果。
- [ ] T010 新增 `riichi_ppo_v1/docs/v18_grp.md` 协议文档与
  `audit/reports/v18/design/V18-GRP 输入扩展设计.md`;
  `audit/reports/v18/report/PROGRESS.md` 追加实施记录;`git diff --check` 通过。
