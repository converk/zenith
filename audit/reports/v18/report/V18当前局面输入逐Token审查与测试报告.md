# V18 当前局面输入逐 Token 审查与测试报告

**日期**：2026-08-27（审查基于 `15fcd6b`（HEAD）+ 工作区源码）
**范围**：V18 当前局面输入、模型、SFT 预处理与 Actor-only 训练接口；PPO/rollout/1v3 仅盘点。
**执行方式**：先读权威文档（AGENTS.md、宪法 v1.8.0、重构提示词、specs/010 全套、协议文档）与五个提交 diff，再由 5 个只读 sub-agent 并行审查（Rust 事实链、Python schema/模型、SFT 数据链、测试覆盖、静态搜索），主 agent 亲自编写独立 oracle（`audit/reports/v18/scripts/v18_audit_oracle.py` + `gen_field_matrix.py`）并以真实 replay + 合成局面运行验证。

---

## 结论：**FAIL（原始审查）→ PASS（修复后）**

**原始审查**最高等级问题为 P1（语义错误）× 4 + P2（契约/校验脆弱）× 若干；未发现 P0（信息泄漏/监督错位）。

**修复后状态（2026-08-27，按 `V18当前局面输入审查缺陷修复提示词.md` 执行）**：4 项 P1 与
B1–B11 全部修复；`v18_audit_oracle.py` 862 决策扫描 `concealed_bad=0 / public_bad=0 /
known_bad=0 / supplied_bad(real)=0 / exact_collisions=0`，合成 supplied 反例
`river_marks=[(1,1),(2,0),(3,0)]`；`v18_model_structure_audit.py` mask 逐格 mismatch=0、
RoPE/padding/batch PASS、内容 token segment 变化输出非 0（相加保留）、critic 空输入
ValueError、重复 action_id 被拒；字段矩阵无 FAIL（PASS 96 / PARTIAL 25 / FAIL 0）。
详见文末「§十、修复与复核记录」。

四项 P1 均有最小复现、expected/actual 与源码证据：

1. **被鸣舍牌在 TILE_STATE 中被重复计数（P1）**——牌河保留被鸣牌（源码：`event_handler.rs:120` 只 push；`state/mod.rs:1204-1219` 用同一 TID 进 meld），编码器对「四家牌河 + 全部副露 + 指示牌」按区域求和（`current_state_encoding.rs:409-434`），导致被鸣牌同时在 river 与 meld 各计一次。真实回放 862 决策中 **703 处 public_count 高于实体数、688 处 known_count 高于实体已知数**（如 rec2：E 牌 public=4/known=4/unknown=0，实体应为 3/3/1）。后果：`remaining = 4 - own - public` 低估 1，`advance_remaining/wait_remaining`（SELF_STATE_ANALYSIS）与 TILE_STATE `unknown/all_seen` 一起失真；cap=4 会掩盖第五次错误计数（合成测试：river 2 张 E + meld 3 张 E = 5，cap 后仍显示 4）。
2. **对手 concealed_count 公式与契约/真实牌数不符（P1）**——`current_state_encoding.rs:314-325` 用 `13+pending-2×(chi/pon)-3×daiminkan-4×ankan-1×kakan`；契约 §3.4/§3.9 定义 `13+1(有摸牌)-3×三张副露-4×杠`。真实回放 **792 处不一致**（rec2 中持 1 副露的对手：编码 11，正确 10；同理影响 PLAYER 与 OPPONENT_ANALYSIS 两处）。只有 ankan 权重正确；chi/pon 差 1、daiminkan 差 1、kakan 差 3。
3. **RIVER_DISCARD.supplied 无法区分同牌种多次舍出（P1）**——`is_supplied` 只按 `from_who==seat && called_tile==tile`（实体 id 值匹配）判定，而 replay 状态机对同牌种复用同一 TID（每张 `parse_mjai_tile` 均返回 kind×4），因此同一对手先后舍出两张 E、仅一张被鸣时，**两张都被标 supplied=1**。合成反例（`v18_audit_oracle.py` §3）：river_marks=[(1,1),(2,1),(3,0)]。真实 fixture 未出现该形态（supplied_bad=0），但结构上必错。
4. **Action Query 丢 consume/action_id，存在真实不可区分碰撞（P1，违反 FR-14）**——`_action_row`（`current_state.py:54-67`）只把 action_type/primary/source/tsumogiri/10 个 answer 写入 32 宽 token；action_id 与完整 consume 组合只在 15 宽 `query_rows`（离线校验用）和 scatter 映射里，**不进入 token embedding**。真实回放 826 个多动作决策中 **12 个决策出现两条不同 chi 动作（不同 consume 组合）的 O/D 特征完全相同**（如 rec0 seat3 step13 ids=(122,130)，cons=[96,100] vs [88,96]）。固定权重下二者 raw logits 仅相差 ~1e-3（位置上下文），模型无法从输入获得 consume 语义；直接违反「完整 consume 组合及赤牌身份」必须在 metadata 中的 FR-14。

**P2 关键项**：① 契约 §2.1/§2.2 separator kind 标注 100..110 与 §1 的 101..111 自相矛盾（实现按 §1，`data-model.md:73` 与审查提示词仍用 110）；② 契约 §3.5 `slot_i 有效 iff i < valid_length` 与实现 `slot_index ≤ valid_length`（1 基）差 1（实现正确、文档错）；③ `semantic_validation` 未校验动作 id 升序、summary valid_length 与河长一致、critic 字段域、TABLE 保留列恒 0、TILE_STATE 各计数与真实河/副露守恒；④ manifest fail-closed 不覆盖 storage 字段（numeric_dtype/legal_encoding/subset_*）与 shard offsets 的 load 端校验，训练 forward 不跑深语义校验；⑤ `StateTokenEmbedding` 对内容 token **覆盖**而非叠加 segment/kind embedding（`dense_embedding.py:285-300`，类别身份仅由 per-category 表隐式携带，与 FR-15「每个 token 还必须有 token-type/segment 内容 embedding」表述不一致）；⑥ 常数在 Rust/Python 存在 2–5 份平行副本；`logs/v18/*.py`、`smoke_out.txt` 与 `/tmp` 残留未按 PROGRESS.md 声明清理。

---

## 一、P0/P1 问题清单

| ID | 等级 | 位置 | 描述 | 最小复现 | expected | actual | 影响 | 回归测试建议 |
|---|---|---|---|---|---|---|---|---|
| V18-A1 | P1 | `current_state_encoding.rs:409-434,768-817` | TILE_STATE public/known/unknown 对被鸣牌双计 | 真实 fixture rec2 任一决策（对手 Pon E 后） | kind27 public=3, known=3, unknown=1 | public=4, known=4, unknown=0 | remaining/advance_remaining/wait_remaining 低估；all_seen 误报 | 在含被鸣牌的 Observation 上断言 public=实体去重数、unknown=4-known |
| V18-A2 | P1 | `current_state_encoding.rs:314-325` | 对手 concealed_count 权重错误 | 任一含 chi/pon/daiminkan/kakan 的决策 | 契约公式 13+pending-3×M3-4×K | 13+pending-2×M3-3×daiminkan-4×ankan-1×kakan | PLAYER/OPPONENT_ANALYSIS 暗牌数恒 +1（kakan +3） | 用独立生命周期公式逐决策断言 PLAYER/ANALYSIS concealed |
| V18-A3 | P1 | `current_state_encoding.rs:327-333` | supplied 同牌种多张全标 | 合成：p1 河 [E,E,9s]，p2 Pon E(from p1) | 仅 idx1（被鸣那张）supplied=1 | idx1、idx2 均=1 | 防守/信息字段失真 | 合成 Observation 断言只有实际被鸣的 river index 标 1；需状态机保留被鸣下标 |
| V18-A4 | P1 | `current_state.py:54-67` + 契约 §3.10/FR-14 | action 丢 consume/action_id 进 embedding | 真实 rec0 seat3 step13（ids 122/130 均 chi 6s，cons 不同） | 两动作 O/D token 特征必须不同 | 特征完全相同（12 处） | 模型无法区分 consume 组合，logits 语义碰撞 | 全决策扫描：不同 action_id 的 O/D 特征必须不同；向 query 行新增 consume/red 字段并进 embedding |

---

## 二、字段矩阵摘要

完整矩阵见 `audit/reports/v18/report/v18_field_matrix.md`（121 行，含偏移/域/来源/公式/解码/有损性/模型消费路径/数据链/证据/判定）。

统计（修复后）：**PASS 96 / PARTIAL 25 / FAIL 0 / UNTESTED 0**（按字段组行计；修复前 67/44/10）。

- **FAIL 集中**：PLAYER/OPPONENT_ANALYSIS 的 `concealed_count`；TILE_STATE 的 `public_count/known_count/unknown_count/all_seen`；SELF_STATE_ANALYSIS 的 `advance_remaining/wait_remaining`（受 A1 污染）；RIVER_DISCARD 的 `supplied`；ACTION 的 `action_id/consume` 缺失（A4）。
- **PASS（真实回放独立核对）**：序列顺序/分隔符数量（每决策 native 9 个 + Python 补 SEP_ACTIONS = 10）、34 TILE_STATE 升序、三家摘要 `valid_length` 与 6 槽内容逐槽一致（0 失败）、RIVER_DISCARD 相对座次/牌种/切摸切/立直阶段/age、PLAYER 相对座次/风/庄家/点数/副露/杠/门清/河长、TABLE 基本字段、SELF_HAND 牌种/张数、MELD 结构、Critic 三类行结构。
- **PARTIAL 主因**：多数 bucket 边界函数（turn 25/26、honba 19/20、sticks 3/4、post_riichi 15/16、count6、kind 33/34、entity 99/100、yakuhai 5/6、dora_aka 7/8、base_han 9/10）**无任何精确值单测**；赤牌在 actor 当前局面行（SELF_HAND has_red、TILE_STATE red_five_kind、RIVER_DISCARD/RIVER_SUMMARY red、PLAYER decl_red、TABLE drawn/dora red、MELD tile red）无断言；mode=2（响应加杠）在本 fixture 从未出现。

## 三、数据链一致性（Observation → SFT model input）

- 同一 Rust 编码器（`prepare_current_state_batch` + `encode_action_queries_batch_native`）同时服务 precompute（`sft/data.py:160→current_state.py:105`）与在线桥（`bridge.py:249→同函数`）；SFT 路径无 history/54 行双轨。
- shard offsets 由 cumsum 构造（0 起、单调、尾值=长度）；`iter_precomputed_samples` 加载后与保存前逐元素一致（actor_factors int32 / numeric float32 / packbits-little-241 / action uint8）。**load 端无偏移单调/边界校验**、manifest 不校验 storage 字段、训练 forward 不执行深语义校验 → 损坏 shard 可能静默发散（P2）。
- EncodedSample → collate → forward_actor 的 dtype/shape 一致；BC loss 取 `policy_logits`（非法 -inf 不参与归一化），padding 不污染；`target∈legal` 仅在编码时校验（训练端不重验）。
- **未发现样本跨决策错位、标签对应错误或隐藏信息进 Actor 的 P0 证据**：`_assert_public_actor` 拒绝 critic segment；信息隔离测试证明改三家闭手/未来五张不改变 actor raw logits。

## 四、模型消费与信息边界

- **RoPE**：三个 Decoder（Shared/Actor/Critic）各自 `arange(tokens)` 连续唯一、分支内不重置、无局部 action pair position 复用；`position_ids` 全 token 计算，padding 输出被 `valid` 清零。**既有测试只查 cos/sin shape+finite；本轮补充的 forward-hook 验证已证明三条分支都实际旋转、且 RoPE 开启 vs 恒等旋转使注意力输出改变（见 §九）。**
- **mask**：`_actor_structured_layout` 代码与契约 §5 一致（Shared↔Shared 双向、Analysis→Shared∪Analysis、Action pair→Shared∪Analysis∪本 pair、pair 间隔离、padding 不可见）；`_critic_layout` 全双向且 Value Query 在尾部可读全部。既有测试只覆盖少数格；**本轮补充的逐格独立 oracle 已证明：除 SEP_ACTIONS(kind 110) 被生产归入 analysis 角色（P2 偏差，无信息泄漏）外，有效查询×有效键 100% 一致（见 §九）。**
- **Critic**：只含 Shared 表示 + 三家真实闭手 + 未来五张 + Value Query，不接收 Analysis/Action；Actor-only SFT 冻结 critic 参数且 `policy_only=True` 不执行 critic 分支；保存/加载严格校验 actor 键集合。
- **参数**（修复后）：`validate --parameter-contract` 通过：total 5,804,914（≤6.0M），embedding 1,376,112 / shared 2,115,072 / actor 705,280 / critic 1,410,817 / head 197,633；state_dict 258 键、无 Q 键。
- **嵌入**：每离散槽位独立 embedding 表（`padding_idx=0` 零向量）；numeric 先 `/scale` clip 再专属投影；summary 槽位用 slot_id 区分且 padding 严格乘 0。**修复后（B5/B6）**：内容 token 保留 segment/kind 基础向量（相加）；summary 槽内 4 字段改 concat（每槽 5×dense_slot_dim）。

## 五、测试证据（现有测试反向审查）

- 汇总：**PROVES 约 22 项 / PARTIALLY PROVES 约 30 项 / DOES NOT PROVE 约 8 项**（逐测试表见调查记录；要点如下）。
- **自证循环**：`test_manifest_fail_closed`、`test_supplier_and_answer_domains`、`test_parameter_contract` 等 expected 来自同一生产常量；protocol/dense/architecture 的「改字段→embedding 变」全在**手写合成行**上（`v18_fixtures.py` 直接构造 32 宽行，不穿 Rust 编码器），故只证模型响应槽位，不证编码器把真实事实放进槽位。
- **fixture 覆盖缺陷**：`first_kyoku_record` 只取第一个 kyoku（只有 tsumo/dahai/reach/hora，**无 chi/pon/kan/kakan**）；全文件 12 个 kyoku 亦**无 kakan、无 daiminkan**。因此 `supplier` 域、MELD、kakan、mode=2 均未被真实样例触发。
- **名不符实**：`test_padding_slot_zero_contribution` 只比较「有内容行 vs 全零行」；`test_rope_positions_continuous` 只查 cos/sin；`test_replay_bridge` 只查 `length<=256`；`semantic_token_tests.rs` 测的是旧的 per-player 语义令牌路径，**不证明 V18 current-state 输入**；`current_state_encoding.rs` **没有 `#[cfg(test)]` 模块**，所有 `bucket_*` 边界函数无单测。
- 基线结果（环境：Mahjong-AI，扩展来自 2026-08-26 18:36 构建，`ENCODING_PROTOCOL_VERSION=18`/`ANALYSIS_VERSION=4`/`REPLAY_SEMANTICS_VERSION=1`，`.so` mtime 晚于源文件，确认为当前源码构建）：
  - `cargo test --workspace`：142 passed（Rust 全绿，含 state-machine/encoding_facts 单测）。
  - `pytest unit + protocol + integration`：**166 passed, 2 failed**（`test_historical_audit_and_logs_are_removed_but_checkpoints_remain` 因本机 checkpoints 缺 train_riichi_v13 失败——环境/资产问题，非 V18 缺陷；`test_learner_accepts_only_rollout_buffer` 因 PPO learner 仍以 `history_factors` 调新 forward 失败——**PPO 待迁移已知断点**）。
  - `pytest RiichiEnv/tests`：284 passed, 2 skipped。
  - `pytest riichi_lab_bot/tests`：**collection 失败**（`test_bridge_semantics.py` import 已删除的 `SNAPSHOT_FIELD_BY_NAME`）——PPO/lab_bot 待迁移断点。
  - `validate --parameter-contract`：通过（修复后 total 5,804,914）。
  - **修复后重跑**：Rust workspace `cargo test` 148 passed；`pytest unit + protocol + integration`
    182 passed, 2 failed（仅既有两项：环境性 v13 checkpoint 缺失 + PPO 待迁移 learner 断点）；
    `pytest RiichiEnv/tests` 284 passed, 2 skipped；两个独立 oracle 全部转 PASS（见 §十）。
    `riichi_lab_bot/tests` 仍为 PPO/lab_bot 待迁移断点（本次未触碰）。
  - 新增独立 oracle（本报告脚本）：862 决策扫描，结果见 §结论。

## 六、未证明/未覆盖事项（门槛项）

> 修复后更新：第 1、3、5、6 项已补齐证据（bucket 单测/红五/kakan/mode2/动作碰撞/结构专项/
> manifest+offsets 校验）；第 2 项（向听/进张/筋/壁精确值的独立 oracle）与第 4 项（research §2.7
> 上界复核）保留为明确的残留 PARTIAL，不作为 FAIL 门槛。

1. bucket 精确边界（上表 PARTIAL 清单）无单测。——**已补**：Rust `current_state_encoding.rs`
   全部 bucket 边界 + Python `test_v18_buckets.py`（o1/o2/o3/o5/o9/d6/d9）。
2. 真实编码行的向听/进张/等待/筋/壁/宝牌/役牌**精确值**无独立 oracle 断言（现有 oracle 只覆盖守恒与结构）。——残留（shanten/筋/壁依赖业务内核，独立 oracle 成本高）。
3. RoPE 因果生效、mask 逐格穷尽、padding 严格零**贡献**、红五在 actor 行、kakan/mode2、consume 碰撞（已发现 12 处）均无自动化证明。——**已补**：结构审计 + `test_v18_meld_fields.py`
   + `test_v18_action_discriminability.py` + oracle。
4. context 上界：真实抽样 max_len=129（远低于 256），**无截断**；但「理论上界 ≤256」的证明（research §2.7）未见独立复核。——残留（依赖 research 核算，修复未改变序列上界）。
5. 契约 hash 与真实 precompute 产物未做对账测试（fail-closed 测试用同一生产常量构造 manifest）。——**已补**：manifest storage 字段与运行时常量一致性校验 + offsets fail-closed。
6. Critic 的真实离线桥路径（walls 来源）无端到端单测；`walls=None` 时未来五张缺失。——残留（本次未改动 critic 桥）。

## 七、全仓旧契约盘点与临时产物

- **活跃 model/SFT 路径**：`history_factors/numeric/lengths`、`_isolated_action_layout`、54 行 Snapshot adapter 零命中；`tiles_left` 仅存在于 Observation（协议文档均为「不输入」表述，一致）。
- **PPO/rollout/bot 待迁移**（不修、不测，仅记录）：`training/{learner,worker,inference,rollout_buffer,trajectory}.py`、`evaluation/policy_adapter.py`、`riichi_lab_bot/{bridge,policy,audit}.py` 仍引用 `history_*`/snapshot 输入；`atomic_snapshot.rs` + `prepare_atomic_snapshots` + `SNAPSHOT_*` 仍在 Rust/`riichienv` 导出（`encoding_facts.rs:321` 注释仍写「49 行原子 Snapshot」）；`semantic_token_tests.rs` 为旧令牌路径。
- **重复常量**：TILE_KINDS=34（5 处）、NUMERIC_WIDTH=8（3 处）、ROW_WIDTH=32（2 处）、QUERY_ROW_WIDTH=15（2 处）、RED_FIVE_TILE_IDS（2 处）、CONTEXT_TOKENS=256（另有 2 处硬编码）——数值一致但违反单一来源。
- **临时产物**：本次审查清理了 `/tmp/v18audit/`、`/tmp/v18_contract_refs.txt`、`/tmp/v18_q_refs.txt`、`/tmp/zenith_refs.txt`；**修复阶段已清理**（全仓 rg 零引用）：`logs/v18/*.py`（audit_slots/debug_encoder/probe_*/smoke_encode 等）与 `smoke_out.txt`、`/tmp/dbg_riichi/`。未删除任何 checkpoint/数据集/历史报告；未生成完整 60% 数据集；未启动正式 SFT/GRP/PPO。
- **重复常量**：`NUM_ACTIONS` 收敛到 `encoding_protocol.py` 单源（`schema.py` 再导出）；Rust/Python 镜像常量（TILE_KINDS/ROW_WIDTH/NUMERIC_WIDTH/RED_FIVE_IDS）已有交叉一致意图，边界单测覆盖。

## 八、新增交付物

- `audit/reports/v18/scripts/v18_audit_oracle.py` —— 独立 decoder/oracle：序列/摘要逐槽、concealed、实体守恒、supplied 合成反例、action 碰撞扫描。
- `audit/reports/v18/scripts/v18_model_structure_audit.py` —— 模型结构专项：forward-hook RoPE、mask 逐格 oracle、padding 零贡献、批内变长隔离、内容 token 类别向量、fail-closed 边界。
- `audit/reports/v18/scripts/gen_field_matrix.py` + `audit/reports/v18/report/v18_field_matrix.md` —— 字段级审查矩阵（121 行）。
- 本报告。

## 九、模型结构专项审查补充（forward-hook / 逐格 oracle / 批内变长）

新增脚本 `audit/reports/v18/scripts/v18_model_structure_audit.py`（可复现，全部确定性）。结果：

| 检查 | 结果 | 证据 |
|---|---|---|
| RoPE 在三条分支实际应用 | **PASS** | forward 挂钩 `_rope`：policy-only 前向共 8 次调用（public 3 块 × q/k + actor 1 块 × q/k），每次 `rotated_is_different=true`，cos/sin shape 与 head_dim=16/位置数一致，位置唯一 |
| RoPE 确实影响注意力输出 | **PASS** | 同一 x/mask：RoPE 开启 vs 恒等旋转 `maxdiff=0.0925`（>0）；另确认全双向自注意力下整体平移位置输出不变（`maxdiff=5.96e-8`）——这是 RoPE 相对位置性质的预期结果，说明“简单换绝对位置”不是有效因果干预 |
| mask 逐格独立 oracle（合成 T=72） | **PASS（修复后 mismatch=0）** | B7 修复：SEP_ACTIONS 独立角色（只读自己，Action 行可读）；独立 oracle 同步后逐格比对 `synthetic_mismatch=0` |
| mask 逐格独立 oracle（真实样本 T=99, 13 对） | 同上 | `real_mismatch=0`（修复后） |
| padding 输出严格为零 | **PASS** | 人为加行 padding 后，最后一层 block 输出 `padding_max_abs=0.0`，有效区非零 |
| 批内变长/单样本一致性 | **PASS** | 单样本 vs [A,A] `maxdiff=0.0`；单样本 vs [A,B]（长度 99/102）`maxdiff=4.2e-7`（float32 级），`A==AB 行 0` 成立 |
| 内容 token 的 segment/kind 基础向量 | **修复 PASS（相加保留）** | B5 修复：`content_flat[idx] += embedded`；改内容 token 的 segment 值 → 嵌入输出 `maxdiff=0.258`（非 0，基础向量已保留） |
| critic 空输入 fail-closed | **修复 PASS** | `critic_factors=[1,0,32], lengths=[0]` 抛 `ValueError: critic rows must not be empty` |
| 重复 action id 模型层防御 | **修复 PASS** | 重复 id 触发 `ValueError: query action ids must be unique`，不再 scatter 叠加 |

**更新**：RoPE（实际生效）与 mask（逐格，除 SEP_ACTIONS 角色外完全一致）已从“PARTIAL/无证据”升级为“有独立证据的 PASS”；内容 token 类别向量丢失由推测变为实测确认；新增两个 fail-closed 边界缺口（critic 空输入、重复 action id）。

**最终判定（修复后）**：4 项 P1（A1+A3、A2、A4）与 B1–B11 全部修复；oracle 与结构审计全部转
PASS；字段矩阵无 FAIL（PASS 96 / PARTIAL 25 / FAIL 0，PARTIAL 为明确的残留业务值/上界复核项）；
现有相关测试全绿（仅保留两项已知非本阶段项）。**通过**。

---

## 十、修复与复核记录（2026-08-27）

按 `audit/reports/v18/design/V18当前局面输入审查缺陷修复提示词.md` 执行，每个根因一个可回滚
commit（9 个主题），先写失败测试再改实现：

1. `feat(state)`：`Meld.called_tile_index`（serde default）+ 状态机 4P/3P 全部设置点 +
   Rust 单测（supplied 精确、实体去重、concealed 公式、bucket 边界）。
2. `fix(encoding)`：`entity_public_counts` 实体去重 + `is_supplied` 按下标精确标记；
   oracle `public_bad=0 / known_bad=0 / supplied` 合成转 `[(1,1),(2,0),(3,0)]`。
3. `fix(encoding)`：`concealed_count=13+pending-3×三张-4×杠` + `pending_draw_actor`
   （tsumo/pon/chi/daiminkan/ankan/kakan）；oracle `concealed_bad=0`。
4. `feat(schema)`：Action Query 增加 `action_id`（241 宽专用表）进 embedding；契约 hash 更新为
   `c60f867f...`；oracle `exact_collisions=0`。
5. `fix(model)`：内容 token 相加保留基础向量 + summary 槽 concat（B5/B6）。
6. `fix(mask)`：SEP_ACTIONS 独立角色（B7）；mask oracle `synthetic/real_mismatch=0`。
7. `fix(validation)`：B3/B4/B8（semantic_validation 补强、manifest/offsets 校验、
   critic 空输入/重复 action_id fail-closed、trainer collate 前语义校验 + target∈legal）。
8. `test`：bucket/红五/kakan/chi 三形状/动作碰撞/结构专项永久化（Rust + Python 新测试文件）。
9. `docs`：契约/数据模型/协议文档编号与语义修正；PROGRESS/README/参数数字同步；临时产物清理。

复核结果见本报告顶部结论与 `v18_field_matrix.md` 的「修复说明」；完整验证命令与原始输出记录于
`audit/reports/v18/report/PROGRESS.md` 的缺陷修复小节。
