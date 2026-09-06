# 项目事实

- 默认使用 `CUDA_DEVICE=0,1` 和 `learner_gpus=2` 进行性能与训练测试;编号 `3`
  对应物理 GPU 4。仅当显式要求单卡运行时才使用 `CUDA_DEVICE=0`。训练入口在
  启动 PyTorch 或 Ray 前,会把 `CUDA_DEVICE` 映射为 CUDA 标准的
  `CUDA_VISIBLE_DEVICES`。
- `CUDA_DEVICE` 映射:`0`→物理 GPU 0,`1`→GPU 1,`2`→GPU 3,`3`→GPU 4。
- 所有 Python 命令与训练使用名为 `Mahjong-AI` 的 Conda 环境。
- `RiichiEnv` 是本项目的训练环境;`RiichiEnv/riichienv-state-machine/` 是 MJAI
  协议状态转换与持久化子包,公开模块名保持 `riichi`,且不依赖 `riichienv`。
- `riichi_ppo_v1/` 是本项目的主训练代码框架。
- 无需与 `evaluations/` 兼容;该组件将被整体重写。
- 默认的性能与训练测试必须显式使用 `target_kl=0.0`、`update_epochs=4`;基线为
  每 update 收集 512 个完整半庄(`games_per_update=512`)。该测试基线与长期训练的
  默认值相互独立。
- 运行测试时默认打印耗时监控和所有相关性能指标。默认运行三轮;将第一轮视为
  潜在预热,并单独报告后续轮次的性能统计。

# 版本契约(最高工程约定)

- 本文件是项目工程约定的最高治理文档;与其他文档冲突时,以本文件为准,并回改
  其余文档保持同步。
- 现行版本契约:encoding protocol 与活跃训练代均为 V19,全仓只保留一个活跃的
  输入、schema、模型与 checkpoint 契约。历史代次的 checkpoint、数据集、日志和
  报告若仍在磁盘上,仅作冷存储,不得被活跃代码加载或迁移,也不得为其保留协议
  适配、旧输入转换、双模型分支、legacy adapter、旧字段 fallback 或 state-dict
  迁移。
- V19 输入协议为**当前局面状态快照 + 信念注入**(Shared 公共前缀 + 三家
  Opponent Analysis + 三家 RIICHI_CARD + 模型内部 30 个信念 token + 每个合法动作
  一对 Offense/Defense Query;全 token RoPE、公共双向 GQA、结构化 Actor mask;
  `d_model=256`/16Q/4KV GQA/`dense_slot_dim=32`/`dense_fusion_dim=512`/
  `context_tokens=320`、`layers=5`(shared 3 + actor 2)/`critic_layers=1`);
  信念网络五头监督与 PPO/SFT 联合训练,标签只进训练不进推理;MJAI 事件仅用于
  同步/生命周期/动作执行,不再作为模型输入。信念分支为 1 层与 Critic 同构的
  backbone(完整 shared_hidden + 每玩家 3 查询共 9 个)+ 五头逐查询平均 +
  逐动作信念读出(SFT detach/PPO 不 detach),30 个信念 token/输入契约不变。
- V19 Actor 的公开信息边界与 Critic 的私有信息边界属于协议契约:任何扩大 Actor
  可见信息、改变 Critic 私有输入或恢复 Q scorer/Q-boosting 的变更,必须先以显式
  规范文档登记并同步协议文档,不得直接修改代码。
- 禁止为旧 checkpoint 长期保留双版本兼容代码;确需新增兼容层时,必须先行显式
  说明并获批准。
- 每个版本的配置必须自包含地写在自己的文件中,禁止 overlay/继承式覆盖;resume
  类配置同样必须是完整自包含副本(按现行 V19 配置的写法)。

# 目录与组件职责

顶层组件:

- `riichi_ppo_v1/` — 主训练框架。
- `RiichiEnv/` — 训练环境(Rust core + PyO3 绑定 + MJAI 状态机)。
- `riichi_lab_bot/` — 在线对局机器人(独立包,`src/` 布局)。
- `docs/` — 跨组件文档(含 `directory-responsibilities.md` 目录职责清单)。

`riichi_ppo_v1/` 包内布局:

- `model/` — 模型、schema 与契约校验
- `training/` — PPO 训练(含 `rewards/` 子包)
- `sft/` — 数据准备、预计算与 SFT 训练
- `evaluation/` — 评测实现与机制常量(1v3 评测的归属目录)
- `tools/` — 工具与生产校验入口(如 `validate.py`)
- `tests/` — unit / integration / protocol 测试
- `configs/` — 版本与默认配置
- `docs/` — 协议与训练文档

文件组织规则:

- 修改代码时保持项目结构整洁。不要仅仅因为某个文件就近,就把不相关的职责堆进
  现有文件;当组件具有独立职责时,应新建文件并放入相应目录,而不是强塞进不相关
  的模块。
- 添加新文件前,先核对当前包布局、命名约定与 `docs/directory-responsibilities.md`;
  相关的实现、测试、文档和配置放在对应目录。
- 禁止跨阶段放置:训练期使用的评测代码不得留在 SFT 目录,生产 CLI 入口不得放在
  tests 包内。
- 代码与文档不得引用已删除的目录或失效路径。

# 产物存储与命名约定

- checkpoint 目录固定为 `checkpoints/train_riichi_<版本号>/`,阶段产物放其下子目录
  (现行为 `train_riichi_v19/sft` 与后续 `train_riichi_v19/ppo`);每个 checkpoint 内部
  保存配置快照以便追溯。
- 现有 checkpoint 一律保留,只允许归档移动,禁止删除(除非另行显式批准)。
- 所有运行日志(json、txt、log 等)写入 `logs/<版本号>/`,禁止在 `logs/` 根目录或
  他处单独生成日志文件。
- `audit/reports/<版本号>/` 是每个版本的初始设计文档、实验报告、测试与验证脚本
  的唯一存放位置,固定子目录:`design/`(设计文档)、`eval/`(评测输出)、
  `report/`(实验报告与 `PROGRESS.md` 进度记录)、`scripts/`(运行与验证脚本);
  禁止随意命名或散落其他目录。
- 现行原始数据集为 `datasets/tenhou_sft_2024_2025`;现行活跃编码数据集为
  `datasets/tenhou_sft_2024_2025_encoded_60pct_v19`（用户已确认的 V19 标准训练输入,
  已预处理完成,禁止重跑 precompute）。`datasets/tenhou_sft_2024_2025_encoded_100pct_v19`
  为全量版本备选/归档。历史代次编码数据与 GRP 数据只允许只读统计,不得覆盖或
  作为活跃训练输入。
- 文件名必须自描述,能从名称看出职责与版本;CLI 默认路径与 README/docs 必须与
  实际产物路径一致。

# 代码与配置约定

- 代码注释一律使用中文(新增与修改的代码必须遵守)。
- 通用性优先:代码不得硬编码实验版本、checkpoint 名、数据集名、对手模型、
  schema/契约 ID、种子基数、计数与间隔、默认路径等,一律通过 CLI 参数或配置项
  传入,默认值不得锁定任何历史版本。
- 允许的例外:与领域本身绑定、不随版本变化的常量(如 136 TID、34 类牌、241 维
  动作空间);这类常量必须收敛为单一命名常量、单一来源,禁止散落各处。
- 新版本迭代通过参数与配置切换实现,不得以复制粘贴代码文件并加版本后缀的方式
  派生实现。
- 评测与验证的节奏参数单点定义:PPO 的 1v3 机制常量在
  `riichi_ppo_v1/evaluation/mechanism.py`,SFT 的节奏在
  `riichi_ppo_v1/sft/contract.py`,禁止在实验配置中复制。
- 删除任何模块或文件前必须执行全仓库引用检查(`rg`);零引用且测试通过才允许删除。
- 重构与清理以"每主题一个 commit、测试通过、可独立回滚"为单位。
- 冒烟测试结束时必须删除其产生的日志与结果文件。

# 评测与验证机制

- PPO 唯一评测机制为 1v3 对抗:固定 10 进程 × 600 = 6000 hanchan(双卡各
  5 进程),每 5 updates 一次;对手模型、种子基数、设备与输出目录由版本配置
  提供,不得硬编码具体版本;输出到 `audit/reports/<版本号>/eval`,进度与失败
  记录写 `audit/reports/<版本号>/report/PROGRESS.md`。
- SFT 的验证与 checkpoint 保存统一固定为每 3000 steps 一次,最终评估规模为
  96 hanchan。
- 上述机制常量(进程数、单进程半庄数、间隔、SFT 节奏)的任何改动,必须先更新
  本文件的机制描述与 `mechanism.py`/`contract.py` 的文档说明,再改代码;禁止在
  单个实验配置里悄悄修改。

# 测试与质量门槛

- 新增或修改代码必须附带对应测试;交付前测试通过。
- 文档(README、docs、AGENTS.md)必须与代码路径同步;协议契约变更必须同步更新
  `KyokuEventTupleProtocol.md` 等协议文档。

# 运行与运维

- 标准训练启动(前台、任意目录、日志落盘):`env -C /mnt/disk1/hubowen/zenith
  RAY_LOG_TO_STDERR=0 CUDA_DEVICE=0,1 PYTHONUNBUFFERED=1 python -m
  riichi_ppo_v1.training.train --config <自包含配置路径> --device cuda
  --learner-gpus 2 2>&1 | tee logs/<版本号>/<运行名>.log`。
- Ray 运维:启动时显式 `RAY_LOG_TO_STDERR=0` 可让组件日志落盘、终端保持干净;
  结束后用 `ray stop --force` 清理,`/tmp/ray/session_*` 残留会话可删除。
- 重要教训:长时间训练运行期间禁止重构、移动模块或产物目录——运行中的进程按旧
  路径拉起评测分片子进程会全部失败。恢复训练用「`resume: <checkpoint>` +
  `init_model: null`」的完整自包含配置。
