# SFT 多版本兼容结构化清理报告（2026-08-02）

## 一句话结论

V13 现已成为唯一的 SFT 预处理、缓存、训练和精确恢复路径；V11 被收敛为只读、冻结的推理岛，并通过统一 adapter 与 V13 使用同一 RiichiEnv 公平评测协议。

## 修改范围与初始审计

- 修改前分支：`sft`；HEAD：`64344a15613463e904f9fa3bda22e7372901efda`；工作区干净，没有需要避让的用户修改。
- 正式数据：`datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16`，manifest SHA256 为 `01529575cc7371a7ae6507c22ea3b64f90a2f25215bee1c8e712543712b2e347`。
- 正式 V13 checkpoint 只读核验：`best.pt` SHA256 `2544f90ed40fc375ef337555023a6a4949d029e1bf13fd934183cb68f40a22cb`；`latest.pt` SHA256 `dd779f57a437144ca9914b79d203e6dd7a84861be5471d6c549d18cf6c661481`。
- 追踪了 schema 11/12/13、encoder、manifest/checkpoint 字段、head fallback、resume cursor、ablation 和两个 evaluator 的实际调用关系。旧分支同时穿过 data/precompute/bridge/train/decision analysis/evaluation，而不是单一兼容入口。

修改前依赖图：

```text
raw replay ─┬─ schema 11 encoder ─┬─ precompute/loader ─┬─ train/resume
            ├─ schema 12 compat  ─┤                     ├─ aligned ablation
            └─ schema 13 encoder ─┘                     └─ validation
                     │
shared bridge + scattered schema switches + legacy source hashes
                     │
              heuristic / head-to-head
```

## 修改后模块图

```text
                         ┌─ precompute ─ encoded loader ─ train ─ exact resume
raw replay ─ V13 bridge ┤
                         └─ V13PolicyAdapter ─┐
                                             ├─ common evaluator ─ RiichiEnv oracle
V11 checkpoint ─ legacy/v11 frozen island ──┘
                 encoder + model + adapter

contract.py: manifest/runtime boundary
checkpoint.py: exact-resume / weights-only boundary
policy_adapter.py: version dispatch boundary
```

V13 的 `train.py`、`data.py`、`precompute.py` 和 `model/bridge.py` 均不导入 `legacy.v11`。版本判断只在 checkpoint/adapter 边界出现。

## 删除、迁移与保留

删除内容及理由：

- 删除四个 V11/V13 aligned/global/multitask ablation 配置；这些路径不属于正式 V13 训练。
- 删除 SFT precompute/data/train/bridge 中 schema 11、schema 12 和通用 `--token-schema` 分支；未知格式改为 fail closed。
- 删除 V11 raw/precompute/encoded-training 能力，以及旧 checkpoint 的 `rank_steps`、缺失 `training_mode`、隐式 policy-head 推断等 resume fallback。
- 删除 `LEGACY_ENCODER_SOURCE_FILES`、legacy encoder 文件级 SHA 和源码 component SHA；训练核心不再绑定 Git/mtime/源码 hash。
- 删除共享 bridge 中的 V11 feature 组装及无调用者的转换逻辑。

迁移至 `legacy/v11/`：

- 固定 contract `riichi-sft-v11-frozen-1`。
- 与历史 checkpoint 匹配的 `legacy_fixed` 模型装配及严格 weights-only loader。
- 历史 V11 token/candidate/public-summary 排列、legal mask、action decode 所需 encoder。
- `V11PolicyAdapter`。它可以复用真正版本无关的 transformer 基元、action mapping、环境 bridge 和手牌分析，但不能生成缓存、训练或 resume。

V13 保留：完整 feature/replay 语义、241 动作和 legal mask、actor-only isolated action head、policy/group/rule 三项 loss、manifest binding、完整 model config/training mode、optimizer/scheduler/RNG、versioned cursor/world-size 校验、validation、heuristic evaluation 和 head-to-head。

## 契约与 checkpoint 行为

新 V13 contract 字段仅为：

- `sft_contract_version`：统一代表 token、feature、replay 时序、numeric 归一化、candidate token 和 241 action mapping。
- `data_plan_version`
- `model_config`
- `training_mode`
- `dataset_manifest_hash`

运行时只另保留一个直接的 `RUNTIME_CONTRACT_ID`，集中核验 Python/Rust replay 分析接口，不再向 checkpoint 扩散多个版本字段。正式旧 V13 缓存和 checkpoint 因禁止改写，只有其精确、已知的历史字段组合可在只读 weights-only/loader 边界通过；任何其他组合均拒绝。

| 行为 | exact resume | weights-only |
|---|---|---|
| 支持版本 | 仅当前 V13 contract | 当前/正式历史 V13；V11 走独立 loader |
| model config/tensor shape | 完全一致、strict | 明确结构、strict shape |
| dataset/training mode/data plan/world size | 必须完全一致 | 不读取训练状态 |
| optimizer/scheduler | 必须存在并恢复 | 不恢复 |
| cursor/per-rank RNG | 必须完整并恢复 | 不恢复 |
| 缺字段、旧 `rank_steps` | 明确拒绝 | 不作为 resume 使用 |

## 统一 adapter 与公平协议

`PolicyAdapter` 暴露 `prepare`、`masked_logits`、`metadata` 和底层只读 `model`；V11/V13 各自封装 feature 语义。公共 evaluator 只处理 observation、legal action、decode、环境推进和指标，不包含 schema 分支。

直接对战按每个 seed 使用同一牌山跑两场并交换两队座位；每个模型覆盖四个座位。异常、超时或非法动作直接失败。输出 paired point delta、标准误、95% CI、胜负/名次和 action-type rates。heuristic evaluator 对两种 adapter 使用相同 seed、cycle、座位轮换、opponent recipe、game mode、max steps 和计分代码。

## 等价性与 golden 结果

V13 修改前/后基线逐字段比较全部 bitwise equal：raw replay 的 factors/numeric/legal/teacher/action/identity、正式 NPZ loader、collate tensors、masked actions。固定 checkpoint raw logits 在 `1e-6` 容差内一致；action 为 `[61, 19, 37, 65]`，token lengths 为 `[57, 62, 70, 75]`。三个 loss 完全一致：policy CE `0.2764755189`、group `0.0`、rule teacher `4.3724679947`。

V11 golden 使用固定 RiichiEnv seed `20260802`：token length `43`、legal count `14`、masked action `58`；top-5 action IDs `[58, 59, 33, 65, 67]`，对应 logits `[6.3799534, 4.9569870, 1.0005530, 0.4798334, 0.03239475]`。测试逐行断言 fixture features，并验证 decode 后动作由 RiichiEnv 接受。

现有测试还覆盖 241 action round-trip、reach 两阶段、chi/pon/kan/hora/kyushu、actor-only、固定 validation identity、bridge、online legal mask/decode 和 DDP data plan。

## 训练与 resume canary

临时 actor-only 数据集含 2048 train/16 validation 样本。双卡命令使用 `CUDA_DEVICE=0,3`、`learner_gpus=2`、global batch `512`，固定总 scheduler horizon 为 3 steps。比较 uninterrupted 3 steps 与 `1 → resume → 2 → resume → 3`：identity sequence、model、optimizer、scheduler、per-rank RNG、data cursor、epoch/global step 的 mismatch 列表均为空；三步 loss 分别为 `2.61983 / 2.43252 / 2.32529`。

该 canary 总耗时 `2.7568s`，处理 1536 samples，`557.16 samples/s`，effective tokens `159018`，padded tokens `238080`，padding fraction `0.33208`，peak GPU `1910.27 MiB`。world-size mismatch、旧 `rank_steps`、缺失 mode、V11 exact resume 和非空 fresh output 均由测试验证拒绝；V11 evaluation loader 成功。

## Evaluation smoke

同一设置各运行两次，去除 elapsed/device 字段后结果一致：

- V11 vs V13 direct：2 半庄（1 个 paired seed、换座同牌山）。双方各赢一场；V11 paired point delta `+21700`，SE `0`，95% CI `[21700, 21700]`。样本仅用于协议/确定性 smoke，不代表模型强弱。
- V11 vs heuristics：2 半庄，point delta mean `-1950`，mean rank `3.0`，first `0`，top2 `0`，kyoku win/deal-in `0.210526/0.157895`。
- V13 vs heuristics：2 半庄，point delta mean `+8400`，mean rank `1.5`，first `0.5`，top2 `1.0`，kyoku win/deal-in `0.291667/0.125`。
- action 使用率由两种评测完整输出；direct smoke 中 V11/V13 的 discard 为 `0.77586/0.76733`、pass 为 `0.16872/0.17698`、reach 为 `0.01724/0.01733`。

## 测试命令与结果

| 命令 | 结果 | 耗时 |
|---|---:|---:|
| `conda run -n Mahjong-AI pytest -q riichi_ppo_v1/tests` | 195 passed, 2 warnings | 10.57s |
| `conda run -n Mahjong-AI pytest -q RiichiEnv/tests` | 282 passed, 2 skipped | 0.51s |
| `conda run -n Mahjong-AI bash -c 'LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" cargo test --workspace'`（`riichi/`） | 6 passed | 1.85s |
| `cargo test --workspace`（`RiichiEnv/`） | 117 passed | 完成 |
| V11/V13 golden + contract 定向测试 | 6 passed | 2.84s |
| `git diff --check` | passed | — |

Rust 只有既有 compiler warnings，无失败。

## 性能前后对比

默认按三次 iteration，第一轮 warm-up，GPU 为 `CUDA_DEVICE=0,3`；SFT 不存在 `target_kl`、`update_epochs`、`kyokus_per_worker`，均标为不适用。

| 指标（warm-up 后两轮聚合/均值） | 修改前 | 修改后 |
|---|---:|---:|
| elapsed | 0.4007s | 0.4092s |
| global samples/s | 2555.47 | 2502.18 |
| per-rank samples/s | 1277.74 | 1251.09 |
| effective tokens/s | — | 423559.20 |
| padding fraction | 0.0307 | 0.0307 |
| forward | 177.993ms | 174.861ms |
| backward | 11.919ms | 117.810ms |
| optimizer | 0.273ms | 0.276ms |
| loader/header scan | — | 24.799ms |
| peak GPU（两测量轮最高） | — | 793.9 MiB |

修改后单轮 samples/s 为 `1389.42 / 12566.41`，共享 GPU 抖动明显；聚合吞吐较旧审计约 `-2.1%`，在该方差下不能判定为回归。语义和 padding fraction 未变。

## 未覆盖项与残余风险

- 本次只执行 2 半庄的协议 smoke，没有执行昂贵的 V11/V13 各 96 半庄两次 heuristic 正式评测，也没有执行建议的 384+ 半庄 direct comparison；因此 smoke 数字不可用于模型排名。入口已支持扩展。
- 没有重扫或重生成正式 V13 缓存，以遵守只读要求；沿用既有全量审计，并以 manifest hash、固定正式 NPZ identities 和 golden batch 核验不变性。
- V11 frozen encoder 仍调用共享 `DecisionAnalysisBatch.build_legacy_v11` 以复用版本无关的牌型分析和环境状态；这个单一兼容 hook 由测试限制只能被 V11 岛调用。历史模型壳和 action mapping 同样复用共享、版本无关实现。
- 历史正式 V13 checkpoint 缺少新 exact-resume 字段，故只能 weights-only；只有此次代码新保存的完整 contract checkpoint 才支持严格精确恢复。

## Git diff summary

清理主体的 tracked diff 为 28 files、`+439/-919`；15 个新增文件共 1127 行，其中本报告 162 行、实现/配置/测试/工具 965 行。合计为 `+1566/-919`。新增职责文件包括 V13 contract/checkpoint/policy adapter、V11 frozen island、canary 配置、baseline capture 和 golden tests。

## 验收问题直接回答

- **V13 训练链路是否已经完全脱离 V11？** 是。正式 train/precompute/data/bridge 无 V11 import 或 schema 分支。
- **V11 checkpoint 是否仍能公平评测？** 是。冻结 V11 adapter 可完成自己的 encoding、严格加载、masked inference、decode，并进入与 V13 相同的 paired/heuristic evaluator。
- **V13 正式缓存是否需要重做？** 不需要；manifest 和固定数据输出未变，且本次没有改写缓存。
- **现有 V13 checkpoint 是否被修改？** 没有；SHA256 和原时间戳均保持不变。
- **新 checkpoint 是否可以精确 resume？** 可以；新格式保存完整 contract、data plan、model/mode/dataset binding、optimizer、scheduler、versioned cursor、world size 和逐 rank RNG，并已通过两次连续恢复等价测试。
- **还剩哪些版本维护代码，为什么必须保留？** 只剩 adapter/checkpoint 边界的显式 V11/V13 dispatch、`legacy/v11/` 冻结岛、正式旧 V13 只读 loader tuple，以及共享 decision-analysis 的单一 V11 hook；它们分别用于历史 checkpoint 公平推理、不可改写正式资产读取和准确复现 V11 features。
