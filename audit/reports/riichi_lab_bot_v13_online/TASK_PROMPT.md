# 任务：让 riichi_lab_bot 加载并运行 V13 模型，通过 RiichiLab 线上 validation（严禁排位）

> 本文档是任务提示词的优化版：保留上一版全部步骤、约束与验收要求，只按当前项目实况修正事实、路径和命令。**不得简化任何一步。**

## 〇、相对上一版的关键更新

1. **模型路径**：V13 checkpoint 现在是 `checkpoints/train_riichi_v13_sft/best_heuristic.pt`（已存在，26,153,403 字节）。上一版所说的“仓库根 `best_heuristic.pt` / 仓库里不存在 `checkpoints/train_riichi_v13_sft/`”已经失效，全部默认值、测试、命令统一改为新路径。
2. **tmux 会话与 token**：会话仍为 `rank`，token 已更换为新 JWT。主代理已把新 token 写入仓库外的私有文件 `/mnt/disk1/hubowen/.riichi_rank_env`（权限 0600），并在 `rank` 会话中重新 source（见 §五.7）。token 值不得出现在本仓库、代码、日志或命令参数中。
3. **GPU**：现在可用的显卡是 `CUDA_DEVICE=2,3`（对应 AGENTS.md 命名中的物理 GPU 3/4，均为 NVIDIA L20 46GB，sm_89，支持 BF16）。所有验证命令同步替换；验证统一显式使用 `--dtype fp32`（`auto` 在 L20 上会选 BF16）。
4. **明确授权修改本地 RiichiEnv**：只要能让实现更简洁、更轻量且功能正常，可以修改本地 RiichiEnv（约束与证明要求见 §三.5）。
5. **业务语义编码正确性提升为第一重点**：模型输入的每个 token 的每个槽位语义都必须被正确编码；模型输出的每个动作都必须被正确解码。任何 fallback / withheld 都视为语义失败。
6. **修正环境路径**：本机不存在 `/home/hubowen24/...`。Conda 环境为 `/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI`，工作区为 `/mnt/disk1/hubowen/zenith`。

## 一、任务目标

V13 模型已经训练完成，但 riichi_lab_bot 仍是 V11 形态，无法加载和正确推理 V13 checkpoint。已知原因：bot 的模型定义、schema 判定、默认 checkpoint、CLI 元数据都是 V11；训练时对 RiichiEnv 的改动导致本地与线上环境存在语义差异风险。你的任务是在**不改变 V13 输入契约**的前提下，升级 bot，使其严格加载 V13 checkpoint、正确编码 241 动作空间与全部 MJAI 事件语义、正确解码模型输出动作，并通过 RiichiLab 官方 validation；全程绝不参与 ranked。

**本任务第一重点是业务语义编码正确性**：送入模型的每个 categorical/numeric token 都必须有可证明的业务语义并与训练路径逐 token 一致；模型输出的每个 action id 都必须被正确解码为 MJAI 动作并与服务器 `possible_actions` 逐字段一致后才发送。

## 二、已确认的项目现状（先读这些，再动手，不要凭空假设）

### 1. 目标模型（已实测核对）

- 路径：`checkpoints/train_riichi_v13_sft/best_heuristic.pt`（约 26MB，本工作区唯一真实存在的 V13 checkpoint）。
- 已用 Mahjong-AI 环境直接 `torch.load` 核对元数据：
  - `sft_contract_version = "riichi-sft-v13-1"`
  - `model_config.policy_head_type = "isolated_action_query"`
  - 顶层没有 `token_schema_version` 字段（判定必须走 `sft_contract_version`）
  - `data_plan_version = 1`
- 训练端 `load_v13_weights_only` 同时兼容 `sft_contract_version` 与旧格式 `token_schema_version == 13`（见 `riichi_ppo_v1/sft/checkpoint.py`）。
- checkpoint 是输入资产，只读使用；不得移动、改名、删除、重训或转换。

### 2. 训练端权威实现（唯一语义来源，必须先读）

- `riichi_ppo_v1/model/feature_schema.py`：冻结的 schema-13 特征契约（categorical/numeric 槽位、范围、缩放、N/A 规则、`ENCODED_FORMAT`）。
- `riichi_ppo_v1/model/schema.py`：`TOKEN_SCHEMA_VERSION = 13`，action-query segment 常量。
- `riichi_ppo_v1/model/architecture.py`：isolated_action_query 模型与 `forward_policy`；`forward_policy` 本身 fail closed 校验“查询对必须是后缀、每个合法动作唯一 offense/defense 对、与 legal_mask 一一对应”。
- `riichi_ppo_v1/model/bridge.py`：`BatchedStateBridge.prepare`，是 bot 必须逐 token 对齐的真值（token 组装顺序：base → public summary → state_tokens → candidate_tokens）。
- `riichi_ppo_v1/model/critic_features.py`：`encode_public_summary` 等公开摘要编码，bot 应优先直接复用。
- `riichi_ppo_v1/model/semantic_validation.py`：`assert_actor_token_semantics(factors, numeric, lengths)`，每个决策推理前必须调用。
- `riichi_ppo_v1/training/rewards/decision.py`：`state_tokens`（六个 segment=6 状态行）+ `candidate_tokens`（每合法动作一个 offense/defense 查询对）。
- `riichi_ppo_v1/sft/checkpoint.py`、`sft/policy_adapter.py`、`sft/contract.py`：V13 严格加载、`load_policy_adapter` 分发、contract id 与运行时断言（`riichi.ANALYSIS_VERSION == 4`、`riichienv.REPLAY_SEMANTICS_VERSION == 1`，必须 fail closed）。

### 3. 运行时环境（已实测）

- Mahjong-AI Conda 环境：`/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI`（直接 python：`/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python`，实测 import 全部成功，torch 2.7.1+cu126）。
- `import riichi, riichienv, riichi_ppo_v1, riichi_lab_bot` 实测通过；`riichi.ANALYSIS_VERSION = 4`、`riichienv.REPLAY_SEMANTICS_VERSION = 1`，这两个是运行时契约，任何修改不得破坏。
- `conda run` 或 import torch 偶尔可能长时间无输出；开始前先做 timeout 预检（见 §五.1）。若卡住先排查环境，不要跳过预检或把环境问题误判为代码问题。

### 4. bot 现状（已核对当前代码）

- `riichi_lab_bot/src/riichi_lab_bot/model.py`：只有 schema 11 `legacy_fixed` 实现（`TOKEN_SCHEMA_VERSION=11`、无 `policy_head_type`），与训练端 `architecture.py` 重复且可能漂移。
- `riichi_lab_bot/src/riichi_lab_bot/policy.py`：强制 `token_schema_version == 11`，V13 checkpoint 必然被拒。
- `riichi_lab_bot/src/riichi_lab_bot/cli.py`：`_load_policy` 遥测硬编码 `token_schema_version=11`，默认 checkpoint 指向不存在的 V11 路径（`checkpoints/train_riichi_v11_sft_40pct_2v2_selection/best_heuristic.snapshot.pt`）；`tests/conftest.py` 默认指向不存在的 V10 路径。
- `riichi_lab_bot/src/riichi_lab_bot/bridge.py`：已导入训练端 `Decision` / `DecisionAnalysisBatch` / `EfficiencyAnalyzer` / `PublicStateTracker` 并注入 segment=7 候选 token，但仍有两个关键缺口：
  - 没有调用 `analysis.state_tokens()` 注入六个 segment=6 状态行；
  - token 拼接顺序与训练 `BatchedStateBridge` 不一致（当前为 base→candidate→public，训练为 base→public→state→candidate）。v13 语义校验要求查询对是后缀，当前顺序违反该约束，必须修正并对齐。
- `riichi_lab_bot/src/riichi_lab_bot/features.py`：bot 自研的公开 summary 编码，与训练端 `riichi_ppo_v1/model/critic_features.py` 的 `encode_public_summary` 存在漂移风险，需要逐 token 等价或直接复用训练实现。
- `riichi_lab_bot/tools/verify_candidate_token_drift.py`：docstring/用法注释已过时（docstring 声称“bot 未注入 candidate tokens”，实际代码已注入），需先修正工具本身再作为验收工具。
- `riichi_lab_bot/README.md`：声称“不导入 riichi_ppo_v1”、默认 checkpoint 为 V10、GPU 为 `CUDA_DEVICE=0,3`、BF16 等，均已过时，需同步修正。
- `riichi_lab_bot/src/riichi_lab_bot/client.py`：`VALIDATION_URL = wss://game.riichi.dev/ws/validate`、`RANKED_URL = wss://game.riichi.dev/ws/ranked`；已有 deadline 250ms 安全区、action_ack 分类（accepted/rejected/unparseable/stale/defaulted）、二进制帧/malformed JSON 忽略逻辑；本次任务不得调用 ranked。
- `logs/` 中存在历史 ranked 日志（旧任务产物），本次任务不得新增或复用 ranked 连接；验收时确认本次新增代码/日志/命令中无 ranked 调用。

### 5. 线上参考文档（以官方文档为准）

- https://riichi.dev/docs
- https://riichi.dev/docs/protocol
- https://riichi.dev/docs/local-testing
- https://riichi.dev/docs/validation
- 上游 RiichiEnv 官方远程仓库：https://github.com/smly/RiichiEnv/tree/main（用于对照本地与线上差异）

### 6. 线上 validation 规则（官方确认）

连接 `wss://game.riichi.dev/ws/validate`，服务器安排 3 个内置 tsumogiri bot 打东场，我方坐席 0（代码从 `start_game.id` 读取，0-3 均合法处理）；只要不 chombo、不断线即通过；超时只会被服务器代打默认动作，不会失败；结束收到 `{"type":"validation_result","passed":true}`。

### 7. 环境与资源（已实测核对）

- 本机 GPU（nvidia-smi / CUDA 枚举）：CUDA=0、1、2、3 为 NVIDIA L20 46GB（sm_89，支持 BF16），CUDA=4 为 T400 4GB。
- 本次任务可用设备：**`CUDA_DEVICE=2,3`**（按 AGENTS.md 命名即 CUDA=2 → 物理 GPU 3、CUDA=3 → 物理 GPU 4；实际枚举 2/3 均为 L20）。单卡显式用 `CUDA_DEVICE=2`，双卡用 `CUDA_DEVICE=2,3`。
- 未经允许不得使用 CUDA=0、1、4（含 T400）。
- 所有验证与线上命令显式 `--dtype fp32`（`auto` 在 L20 上会选 BF16）；BF16 只用于“FP32/BF16 推理一致”的契约测试（在 L20 上执行）。
- `CUDA_DEVICE` 会在导入 PyTorch 前映射为 `CUDA_VISIBLE_DEVICES`（见 `bootstrap.py`）。因此 `CUDA_DEVICE=2,3` + `--device cuda:0` 时，进程内 `cuda:0` 是第一个可见设备（即逻辑 CUDA=2 对应的 L20），不是物理索引 0。

## 三、硬性约束（违反任何一条都算任务失败）

1. **严禁排位**：不得运行 `riichi-lab-bot ranked`、不得加 `--forever`、不得以任何方式连接 `wss://game.riichi.dev/ws/ranked`；线上验证只允许 validate。所有子代理同受此约束；最终在本次新增的代码、日志、shell 历史中确认没有 ranked 端点连接或 ranked 子命令调用（`client.py` 中既有 `RANKED_URL`/`ranked` 子命令属既有代码，不在本次修改范围，也不得执行）。

2. **绝不改变 V13 输入契约**：六个 state 行 + 每个合法动作的 offense/defense 查询对 + legal mask + 公开 summary 的 token 布局、categorical/numeric 槽位、数值范围、缩放、顺序、N/A 语义都必须与 `riichi-sft-v13-1` 冻结契约完全一致。不得修改 `feature_schema.py`、模型架构参数、checkpoint 权重；不得重训或转换 checkpoint。

3. **语义必须逐 token 校验（本任务第一重点）**：**模型输入的每个 token 的每个槽位都必须有可证明的业务语义并被正确编码；模型输出的每个动作都必须被正确解码。** 具体要求：
   - 每个决策在推理前（或至少在每个测试中）调用 `riichi_ppo_v1/model/semantic_validation.assert_actor_token_semantics` 校验 factors/numeric/lengths；查询对必须是后缀、offense/defense 相邻、action_id+1 唯一，并与 legal_mask 一一对应。
   - 以训练 `BatchedStateBridge.prepare(decisions, analysis)` 为逐 token 真值：bot 的 `token_factors` / `token_numeric` / `token_length` / `legal_mask` 必须与训练路径逐元素相等，包括六个 state 行、public summary 及拼接顺序（base → public → state → candidate）。
   - 事件语义矩阵必须覆盖：start_game、start_kyoku、tsumo、dahai、chi、pon、daiminkan、ankan、kakan、dora、reach、reach_accepted、hora、ryukyoku、end_kyoku、end_game，以及未知事件/字段的处理；重点核对相对座位、红五（tile id 16/52/88 ↔ 5mr/5pr/5sr）、摸切标记、consumed 归一化（是否包含被叫牌）、dora 指示牌重复倍率、立直宣言牌、和牌 target、本场/供托、分数连续性、对手手牌掩码 `"?"`。未知事件和未知字段必须忽略，不能报错。
   - **动作解码同样属于语义正确性**：模型输出的 action_id 必须经 `decode_actions` 得到 MJAI 动作，再经 `select_action_from_mjai` 与服务器 `possible_actions` 逐字段比对（type、pai、consumed、tsumogiri、必要时 actor/target）；发送 payload 的 canonical JSON 必须等于模型解码动作（只允许附加 request_id、tsumogiri、target 等服务端要求字段）。

4. **动作编码必须严格（本任务第二重点）**：241 个固定 action id 与 MJAI JSON 的双向映射必须与训练完全一致；模型输出先经 `decode_actions` 得到 MJAI 动作，再经 `select_action_from_mjai` 与服务器 `possible_actions` 逐字段比对（type、pai、consumed、tsumogiri、必要时 actor/target），两者都合法才发送。最终验收要求：每个送入环境的动作都必须是模型所选出的动作——本地与线上都必须 `model_actions == 决策数`，且 `fallback_actions = 0`、`withheld_actions = 0`；任何 fallback/withheld 都视为语义失败，必须修复根因，不能靠兜底机制“通过”。

5. **允许修改本地 RiichiEnv（明确授权）**：本任务明确声明，如果修改本地 RiichiEnv 能让整个实现更简洁、更轻量且功能正常，可以修改。但必须同时满足：
   - V13 模型输入的每个组件（六个 state 行、candidate 查询对、legal mask、public summary 及每个 categorical/numeric token）仍能从线上服务器下发的 MJAI 事件和 base64 Observation 中正确重建，且语义与训练时完全一致；
   - 不得破坏训练语义：`ANALYSIS_VERSION == 4` / `REPLAY_SEMANTICS_VERSION == 1` 相关契约必须 fail closed；
   - 每处修改都要配测试和逐项证明；
   - 优先复用训练端 bridge/编码实现，其次才考虑改环境；
   - 修改前先记录本地与线上差异，修改后给出“每个 V13 输入组件仍可从线上 Observation 获得且语义不变”的逐项证明。

6. **模型加载必须严格**：按 `sft_contract_version` / `policy_head_type` / `token_schema_version` 识别 V13 契约；必须 `strict=True` 加载全部权重；不得为了绕过检查而放宽校验、丢弃权重或忽略 `model_config` 字段；FP32 与 BF16 推理结果一致（BF16 测试在 L20 上执行）。

7. **环境与资源**：使用 Mahjong-AI conda 环境（`/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI`）。本任务验证与本地对局使用 `CUDA_DEVICE=2,3`（单卡显式用 `CUDA_DEVICE=2`），dtype 一律显式 `fp32`（BF16 仅用于一致性契约测试）；未经允许不得使用其他 GPU。

8. **修改范围**：只允许修改 `riichi`、`riichi_lab_bot`、`riichi_ppo_v1`、`RiichiEnv` 四个目录；`checkpoints/train_riichi_v13_sft/best_heuristic.pt` 是输入资产，只读使用，不得移动、改名、删除。保持项目结构整洁，独立职责放独立文件。

9. **Token 安全**：`RIICHI_BOT_TOKEN` 只从环境变量读取，绝不写入代码、仓库、日志或命令参数；token 已配置在仓库外的私有文件 `/mnt/disk1/hubowen/.riichi_rank_env`（权限 0600，只含一行 `export RIICHI_BOT_TOKEN='<token>'`），不要提交、复制或打印；包含 token 的输出必须脱敏；校验只能打印字符数。

10. **V11 兼容性**：除非明确决定并说明理由，不得破坏 V11 checkpoint 的加载能力（训练端已有 `legacy/v11` 适配器可复用）。

11. **多智能体授权**：本任务明确允许使用多智能体协作 / 开启子代理（当前环境最多 4 个并发 agent，含主代理）。建议按以下边界并行拆分：差异审计、模型层与加载升级、bridge 语义修复与逐 token 等价测试、边缘业务测试矩阵与文档/默认值更新；子代理通过共享工作区协调，最终由主代理统一验收。所有子代理必须遵守本节全部约束（尤其严禁 ranked、token 安全、模型文件只读）。

## 四、实现路径建议

1. **先做差异审计**：对照官方 protocol 文档和上游 RiichiEnv，列出本地 RiichiEnv 与线上环境在事件、规则、Observation 字段上的差异；列出 bot 从 V11 到 V13 的全部改动点（模型结构、schema 判定、state/candidate tokens、拼接顺序、默认 checkpoint、CLI/README/测试）。

2. **模型层**：直接复用 `riichi_ppo_v1/model/architecture.py` 与 `sft/checkpoint.load_v13_weights_only`（或 `sft/policy_adapter.load_policy_adapter`）作为唯一权威实现；把 bot 的 `model.py` 收敛为对训练端的再导出或直接删除，禁止保留两套可能漂移的模型定义。

3. **加载与契约判定**：`policy.py` 改为按 `sft_contract_version` / `policy_head_type` / `token_schema_version` 判定（V13 checkpoint 顶层无 `token_schema_version`，必须支持 `sft_contract_version == "riichi-sft-v13-1"` + `model_config.policy_head_type == "isolated_action_query"`）；`cli.py` 的 `token_schema_version` 元数据改为从 checkpoint 实际值读取；默认 checkpoint 与 `tests/conftest.py` 都指向 `checkpoints/train_riichi_v13_sft/best_heuristic.pt`。

4. **bridge 语义**：按训练路径的精确顺序组装 token：base（Rust 状态机 history/state）→ public summary → `state_tokens`（六个 segment=6 行）→ `candidate_tokens`（segment=7 查询对）；每步用 `assert_actor_token_semantics` 校验；确保在线事件 → riichienv.Observation → riichi 状态机 → schema-13 token 的链路与训练路径逐 token 一致。公开 summary 优先复用训练端 `critic_features.encode_public_summary`，或保留 bot 实现但必须逐 token 等价。

5. **动作发送路径**：模型 argmax → `decode_actions` → `select_action_from_mjai` → 与 `possible_actions` 比对 → 发送。发送 payload 的 canonical JSON（`json.dumps(..., separators=(",", ":"), sort_keys=True)`）必须等于模型解码动作（只允许附加 request_id、tsumogiri、target 等服务端要求字段）；任何一步失败都不得用 fallback 顶替，必须暴露并修复。

6. **边缘业务测试矩阵（业务语义优先，必须覆盖这些犄角旮旯）**：
   - 九种九牌/流局（ryukyoku，action id 240）、抢杠（kakan 后和牌）、岭上、海底/河底、多家和；
   - 红五：5mr/5pr/5sr 与物理 tile id 16/52/88 的映射，普通五与红五不混淆；chi/pon consumed 中的红五组合；
   - 摸切标记 tsumogiri true/false；立直后 tile=None 的 follow-up dahai；
   - consumed 归一化：chi/pon/daiminkan 三态分别验证“被叫牌是否包含在 consumed 中”的两种来源；
   - dora 指示牌：重复指示牌按倍率累计，kakan/dora 事件后的指示牌变化；
   - reach 宣言牌、reach_accepted、本场/供托、分数连续性、end_game scores；
   - 相对座位 1-4、start_game seat 0-3、多局后状态机 reset（end_kyoku/end_game → 下一局）；
   - 对手手牌掩码 `"?"`；未知事件/未知字段/二进制帧/malformed JSON 一律忽略不崩溃；
   - deadline 边界：接近 deadline 不发送可能迟到的动作；action_ack 的 rejected/unparseable/stale/defaulted 分类计数；
   - 长局/多副露下 token context 接近 4096 的边界；seat mismatch / 非法 request_action。

7. **更新测试与文档**：conftest 默认 checkpoint、CLI 默认 checkpoint、README 的 schema/依赖/GPU/路径描述、`verify_candidate_token_drift.py` 的注释和用法、测试断言。

## 五、验证阶梯与验收命令（按顺序执行，每一步通过才进入下一步）

### 1. 环境与运行时检查

```bash
cd /mnt/disk1/hubowen/zenith
timeout 60 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -c "import riichi, riichienv, riichi_ppo_v1, riichi_lab_bot; print('runtime ok')"
timeout 60 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -c "import riichi, riichienv; assert getattr(riichi, 'ANALYSIS_VERSION', None) == 4; assert getattr(riichienv, 'REPLAY_SEMANTICS_VERSION', None) == 1; print('runtime contract ok')"
test -f /mnt/disk1/hubowen/zenith/checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

如果 conda run 或 import torch 卡住，先修复/说明环境问题再继续（可直接使用 `/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python`，不要依赖 conda activate）。

### 2. 全量离线测试

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest riichi_ppo_v1/tests riichi_lab_bot/tests RiichiEnv/tests -q
```

（`conda run -n Mahjong-AI ...` 卡住时同样改用上面的直接 python 路径。）

重点测试必须全部通过：
- `riichi_ppo_v1/tests/protocol/test_action_space_exhaustive.py`、`test_protocol_matrix.py`
- `riichi_ppo_v1/tests/unit/test_feature_schema_v13.py`、`test_semantic_validation.py`
- `riichi_ppo_v1/tests/integration/test_v13_sft_golden.py`、`test_v11_policy_adapter.py`、`test_bridge_integration.py`、`test_real_action_cases.py`
- `riichi_lab_bot/tests/test_checkpoint.py`、`test_bridge_integration.py`、`test_safety.py`、`test_client.py`

必须新增/更新：
- V13 加载测试：`strict=True` 成功、BF16 与 FP32 推理一致、错误契约/错误 `policy_head_type` 被拒；
- 语义逐 token 测试：每个决策 `assert_actor_token_semantics` 通过，查询对后缀/唯一/与 legal_mask 一一对应；
- bot bridge 与训练 bridge 逐 token 等价（含六个 state 行、public summary 顺序、legal_mask）；
- 发送动作等价测试：发送 payload 的 canonical JSON == 模型解码动作；
- §四.6 的完整边缘业务语义矩阵。

### 3. 语义与动作空间覆盖

```bash
cd /mnt/disk1/hubowen/zenith
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-ppo-validate --games 128 --seed 20260713 --max-steps 2500 --output riichi_ppo_v1_coverage.json
```

可加跑 256/512 场；检查事件类型、动作组的覆盖情况。

### 4. bot bridge 与训练 bridge 逐 token 等价

```bash
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python -m pytest riichi_lab_bot/tests/test_bridge_integration.py -q
```

要求：同一决策下 bot 的 `token_factors`、`token_numeric`、`token_length`、`legal_mask` 与训练 `BatchedStateBridge` 逐元素相等；并断言每个合法动作都有唯一且相邻的 offense/defense 查询对。

### 5. 候选 token 漂移检查

```bash
cd /mnt/disk1/hubowen/zenith
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/python riichi_lab_bot/tools/verify_candidate_token_drift.py --model checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

要求 `disagreements = 0`；工具内注释/用法已过时（docstring 仍声称“bot 未注入 candidate tokens”，实际代码已注入；默认模型路径仍是 V11），先按实际代码修正工具本身再验收。

### 6. 本地三局测试（warm-up + 两局统计）

```bash
cd /mnt/disk1/hubowen/zenith
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot local \
  --games 3 \
  --seed 20260730 \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

要求：三局完整结束；第 1 局作为 warm-up，第 2、3 局单独报告推理统计（决策数、耗时、decisions_per_second、inference 均值/P50/P95）；严格满足 `model_actions == 决策数`、`fallback_actions = 0`、`withheld_actions = 0`；每局输出 scores/ranks。

另跑一次 CPU 冒烟：

```bash
cd /mnt/disk1/hubowen/zenith
/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot local \
  --games 1 \
  --device cpu \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt
```

### 7. 线上 validation（在 tmux 会话 `rank` 中执行，token 已配置好）

说明：tmux 会话 `rank` 已存在，`RIICHI_BOT_TOKEN` 已配置为新 token。token 保存在仓库外的私有文件 `/mnt/disk1/hubowen/.riichi_rank_env`（权限 0600，只含一行 `export RIICHI_BOT_TOKEN='<token>'`），会话启动时自动 source；该文件不要提交、复制或打印。不要在普通 shell、命令参数或日志中重新读取/输出 token。

使用方法（会话已存在，直接附加）：

```bash
tmux attach -t rank
```

会话内已把 `/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin` 加入 PATH，因此 `python`、`pytest`、`riichi-lab-bot` 都指向 Mahjong-AI 环境（不依赖 conda activate）。

如果之后在会话里新开窗口/面板导致环境变量丢失，先执行：

```bash
source /mnt/disk1/hubowen/.riichi_rank_env
export PATH="/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin:$PATH"
```

重新设置 token 的方式（token 只写入私有文件，绝不出现在命令参数或日志里）：

```bash
# 1) 用用户提供的新 JWT 覆盖仓库外私有文件：
#    /mnt/disk1/hubowen/.riichi_rank_env 内容为：
#    export RIICHI_BOT_TOKEN='<用户提供的新 JWT>'
chmod 600 /mnt/disk1/hubowen/.riichi_rank_env
# 2) 在新 tmux 会话中 source（若 rank 不存在则重建）：
tmux new-session -d -s rank -n rank \
  'source /mnt/disk1/hubowen/.riichi_rank_env; export PATH="/mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin:$PATH"; cd /mnt/disk1/hubowen/zenith; exec bash'
tmux attach -t rank
```

不打印 token 的校验方式：

```bash
test -n "$RIICHI_BOT_TOKEN" && echo "token ok: ${#RIICHI_BOT_TOKEN} chars"
```

在 `rank` 会话内、工作目录 `/mnt/disk1/hubowen/zenith` 下执行验证：

```bash
cd /mnt/disk1/hubowen/zenith
CUDA_DEVICE=2,3 /mnt/disk1/hubowen/miniconda3/envs/Mahjong-AI/bin/riichi-lab-bot validate \
  --device cuda:0 \
  --dtype fp32 \
  --checkpoint checkpoints/train_riichi_v13_sft/best_heuristic.pt \
  --jsonl-log logs/validate_v13_$(date +%Y%m%d_%H%M%S).jsonl
```

（若仍需 `conda run` 形式：`CUDA_DEVICE=2,3 conda run -n Mahjong-AI riichi-lab-bot validate ...`，卡住时用上面直接入口脚本的等价命令。）

要求：退出码 0，JSON 输出 `validation_passed: true`（对应官方 `validation_result.passed`）。失败（退出码 2）时用 `--jsonl-log` 留证并分析原因；可重试最多 3 次独立 validation；任何情况下都不得改用 ranked。若失败源于本地环境与线上规则差异，先用 `RiichiEnv/scripts/validate_logs.py` 或针对性测试复现，修复后从第 2 步重新回归。

### 8. 性能基线（如需要）

按 AGENTS.md 的默认基线跑 3 轮，第 1 轮 warm-up，单独报告第 2、3 轮；使用 `target_kl=0.0`、`update_epochs=4`、`kyokus_per_worker=1`。AGENTS.md 写的 `CUDA_DEVICE=0,3` 是项目文档默认，但本次任务明确可用设备为 `CUDA_DEVICE=2,3`；如需跑基线先用 `nvidia-smi` 确认可用设备，再按实际使用 `CUDA_DEVICE=2,3`，并在报告中说明与 AGENTS.md 默认值的偏差；本次任务默认不需要训练。

## 六、最终验收标准（全部满足才算完成）

1. §五.2～§五.6 所有离线测试/命令通过，且每个关键输出已留存。
2. V13 checkpoint 严格加载（`sft_contract_version = "riichi-sft-v13-1"` / `policy_head_type = "isolated_action_query"` 正确识别，`strict=True`，FP32/BF16 均可）。
3. bot 与训练路径逐 token 等价（含六个 state 行、public summary、拼接顺序），candidate token drift = 0。
4. 本地 3 局 100% model actions：0 fallback、0 withheld，性能统计完整。
5. 线上 validation 至少一次 `passed: true` 且退出码 0。
6. 每个送入环境的动作都证明是模型所选动作（发送日志 canonical JSON 与 decode 输出一致 + 0 fallback/0 withheld）。
7. 全程没有连接 `/ws/ranked`，没有运行 ranked 子命令（含所有子代理；本次新增代码、日志、命令历史中均无 ranked 调用）。
8. 若修改了 RiichiEnv：给出每个 V13 输入组件“仍可从线上 Observation 获得且语义不变”的逐项证明。
9. V11 兼容性影响已说明（保留或明确移除，并同步测试与 README）。

## 七、报告要求

- 列出所有修改的文件、每处修改的原因。
- 附上每个验证步骤的实际命令、关键输出和通过/失败结论。
- 提供“动作空间转义核对表”和“事件语义转义核对表”，覆盖 §三.3/§三.4 列出的每个点。
- 提供“语义逐 token 校验矩阵”和“边缘业务测试矩阵”的实际覆盖结果。
- 说明是否修改了 RiichiEnv，以及本地环境与线上环境的剩余差异和风险；若修改，给出逐项证明。
- 说明子代理/多智能体的使用情况（哪些子任务并行、谁验收）。
- 明确声明：本次工作未连接 ranked 端点。
