# V16 spec-kit 开发提示词

> 本文档是喂给 `$speckit-specify` 的提示词,用于在 V16 分支上按 spec-kit 流程完成
> V16 开发。本提示词只声明**流程与约束**,具体业务逻辑一律以设计文档
> `audit/reports/v16/design/V16 网络结构与训练方案.md` 为权威依据,不在此复述。

## 使用方式

1. 确认当前分支为 `V16`;spec 目录编号由 `.specify/init-options.json` 的
   sequential 规则自动续接,预期为 `specs/003-...`。
2. 将下方「提示词正文」整体作为 `$speckit-specify` 的输入执行。
3. 依次执行 `$speckit-plan` → `$speckit-tasks` → `$speckit-implement`,并按需使用
   `$speckit-clarify`、`$speckit-analyze`、`$speckit-converge`。
4. 建议 feature 短名使用 `v16-model-rework`(最终目录形如
   `specs/003-v16-model-rework`)。

---

## 提示词正文

### 任务背景

在 V16 分支上完成 V16 版本的开发。唯一的业务/技术权威来源是
`audit/reports/v16/design/V16 网络结构与训练方案.md`(以下简称「设计文档」),
包括但不限于:网络结构与参数量、Actor/Critic 输入、统一 Action Query Schema、
策略头融合、Top-3 Q-boosting、GRP 模型与奖励、训练流程等所有具体设计。

训练数据来源为 [tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai)
提供的 **2024 年与 2025 年** Tenhou 对局数据;仓库内现行原始数据目录为
`datasets/tenhou_sft_2024_2025`,编码后数据目录为
`datasets/tenhou_sft_2024_2025_encoded_40pct_v13_v16`。V16 的 SFT 重新编码与
GRP 数据集均从该来源构造,新编码版本生成的新数据集按仓库命名规范版本化存放。

本提示词不规定具体业务逻辑;spec/plan/tasks 阶段必须逐条落地设计文档内容,
除本提示词声明的约束外不得自行改动设计意图。

### 开发流程声明

- 严格按 spec-kit 流程:`$speckit-specify` →(按需 `$speckit-clarify`)→
  `$speckit-plan` → `$speckit-tasks` → `$speckit-implement`,最后用
  `$speckit-analyze` / `$speckit-converge` 校验一致性;
- 三件套(spec/plan/tasks)存于 `specs/<NNN-name>/`,遵循仓库既有模板与命名;
- 设计文档中未写死的实现细节(版本号、bucket 边界、目录归属、删除清单等)由
  spec 阶段拍板并记入 Assumptions,业务语义仍以设计文档为准;
- 每个阶段产物必须符合本仓库目录、文档与代码结构管理规范。

### 约束声明

1. **旧代码清理**:可以随意删除旧版本代码,无需考虑兼容性。删除前必须做全仓库
   `rg` 引用检查,零引用且测试通过才允许删除;按「每主题一个 commit、测试通过、
   可独立回滚」执行。v11 权重与 v14 资产仅冷存储保留,checkpoint 与数据集一律
   不删除。

2. **输入协议与文档**:按设计文档的新编码输入设计,实现新的模型输入转换;同步新增
   或更新模型输入协议文档,明确说明新版本模型输入协议。契约/协议文档必须与代码
   实现同步(如 `KyokuEventTupleProtocol.md` 等)。新 token schema 版本与编码格式
   版本由 spec 明确编号,并同步更新宪法 Principle II 的现行契约声明
   (经 `$speckit-constitution` 修订并记录 Sync Impact Report)。

3. **分析函数归属**:按合适性决定新局况分析函数放
   `RiichiEnv/riichienv-state-machine` 还是 `RiichiEnv/riichienv-core`:
   仅由公开 MJAI 状态与自身信息可确定的事实放 state-machine(公开模块名保持
   `riichi`,不得依赖 `riichienv`);需要规则/手牌结构评价的事实放 core,并优先
   复用既有 shanten/手牌评价/yaku 分析;仅训练侧可组合的事实放模型输入转换侧。
   每项新函数在 spec/plan 中写明归属与理由,不得引入反向依赖。

4. **语义与业务正确性测试(硬性门槛)**:编写完成后必须对所有编码情况做语义与
   业务正确性测试,保证编码进模型的输入与局面上实际发生/计算的事实一致,尤其要
   对每个 action query 的询问逐项验证。测试必须用独立 oracle 重算比对,禁止
   编码器自证;覆盖设计文档定义的全部 slot 语义、N/A 规则、bucket 边界与边界
   局面;验证 Actor 输入不含隐藏信息、特权信息只出现在 Critic;验证所有
   categorical 因子在 cardinality 范围内;并用回放/桥接测试验证环境局面与编码
   tensor 一致。

5. **工程治理**:
   - 目录按职责放置(模型/契约、训练、SFT、评测、工具、测试各自归位),新组件
     建独立目录/子包,不得塞进无关模块;
   - 领域常量(136 TID、34 牌类、241 动作维、bucket 基数等)收敛为单一命名
     常量、单一来源;
   - 配置自包含写在自己的文件,禁止 overlay/继承;版本号、checkpoint、数据集、
     对手模型、schema ID、种子、间隔、路径一律经 CLI/配置传入,默认值不得锁定
     历史版本;
   - 产物按规范:checkpoint 到 `checkpoints/train_riichi_v16/`(含配置快照)、
     运行日志到 `logs/v16/`、报告与脚本到 `audit/reports/v16/`(design/eval/
     report/scripts)、评测输出到 `audit/reports/v16/eval`,进度写
     `audit/reports/v16/report/PROGRESS.md`;
   - 评测机制(PPO 1v3 与 SFT 节奏)沿用现行机制常量单点定义,不得在实验配置中
     复制或悄悄改动;如需改动必须走 `$speckit-constitution` 宪法修订;
   - 性能与训练测试固定 `target_kl=0.0`、`update_epochs=4`、
     `kyokus_per_worker=16`,`CUDA_DEVICE=0,1`、`learner_gpus=2`、Conda 环境
     `Mahjong-AI`;默认跑 3 轮、首轮视为预热并单独报告后两轮,默认打印耗时监控
     与全部相关性能指标,冒烟测试结束删除其产生的日志与结果文件。

### 完成判定

- 全部语义与业务正确性测试通过,任一 action query 的 slot 与独立 oracle 不一致
  即视为失败;
- 新模型输入协议文档与实现一致,schema/编码版本唯一并经宪法修订登记;
- 旧代码零引用清理完成,全仓库测试通过;
- README、docs 与代码路径同步,评测机制未被悄悄改动。
