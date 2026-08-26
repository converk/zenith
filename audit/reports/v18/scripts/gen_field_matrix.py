"""生成 V18 当前局面输入字段级审查矩阵（audit/reports/v18/report/v18_field_matrix.md）。

字段定义来自 specs/010 契约文档 §3（审查基准），不读取生产 CATEGORY_SCHEMAS。
判定列由本审查结论给出：
  PASS=代码/运行证据支持；PARTIAL=有域/结构证据但无独立语义值断言；
  FAIL=发现语义错误（见结论列）；UNTESTED=无任何自动化证据。
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("/mnt/disk1/hubowen/zenith/audit/reports/v18/report/v18_field_matrix.md")

# (kind, 字段名, 偏移, 域, 真值来源, 编码公式, 独立解码, 有损性, 模型消费, 数据链, 证据, 判定, 说明)
ROWS: list[tuple[str, ...]] = [
    # ---- TABLE ----
    ("TABLE", "round_wind", "2", "0=E..3=N", "Observation.round_wind", "原值", "0..3", "无损", "dense 表 slot0", "obs→Rust→共享行", "域校验/真实回放抽样", "PASS", "与起始庄/风一致"),
    ("TABLE", "kyoku_index", "3", "0..3", "Observation.kyoku_index", "原值", "0..3", "无损", "dense 表 slot1", "同上", "同上", "PASS", ""),
    ("TABLE", "honba_bucket", "4", "0..19,20=20+", "Observation.honba", "bucket_honba", "精确或 20+ 等价类", "bucket", "dense 表 slot2", "同上", "无边界单测", "PARTIAL", "bucket 函数无单测"),
    ("TABLE", "riichi_sticks_bucket", "5", "0..3,4=4+", "Observation.riichi_sticks", "bucket_sticks", "精确或 4+ 等价类", "bucket", "dense 表 slot3", "同上", "无边界单测", "PARTIAL", ""),
    ("TABLE", "oya_seat", "6", "0..3", "Observation.oya", "原值", "绝对座次", "无损", "dense 表 slot4", "同上", "真实回放抽样", "PASS", ""),
    ("TABLE", "self_seat", "7", "0..3", "Observation.player_id", "原值", "绝对座次", "无损", "dense 表 slot5", "同上", "真实回放抽样", "PASS", ""),
    ("TABLE", "decision_mode", "8", "0=主动,1=响应弃牌,2=响应加杠", "drawn_tile + new_events 末事件", "drawn_tile.is_some()→0; 末事件 kakan→2, 否则 1", "模式", "无损(依赖事件增量)", "dense 表 slot6", "obs→Rust(读 events)", "真实回放 mode{0:648,1:214}；mode2 无样例", "PARTIAL", "mode=2 依赖事件增量掩码语义，未在真实 kakan 上验证"),
    ("TABLE", "drawn_tile_type", "9", "0=N/A,1..34", "Observation.drawn_tile", "tile_type_code", "牌种或 N/A", "无损", "dense 表 slot7", "同上", "真实回放抽样", "PASS", ""),
    ("TABLE", "drawn_tile_red", "10", "0/1", "drawn_tile ∈ {16,52,88}", "red_flag", "是否赤五", "无损", "dense 表 slot8", "同上", "fixture 无红五抽牌断言", "PARTIAL", "真实数据无红五 draw 样例"),
    ("TABLE", "drawn_is_current", "11", "0/1", "drawn_tile.is_some()", "原值", "=mode0", "无损", "dense 表 slot9", "同上", "真实回放 mode 计数", "PARTIAL", "与 decision_mode 冗余，无独立断言"),
    ("TABLE", "self_riichi_status", "12", "0/1/2", "riichi_declared/accepted", "self_riichi_status", "三态", "无损", "dense 表 slot10", "同上", "真实回放立直记录", "PASS", ""),
    ("TABLE", "dora_indicator_type_slot_1..5", "13..17", "0=N/A,1..34", "Observation.dora_indicators", "按出现顺序", "牌种或 N/A", "无损(上限5)", "dense 表 slot11-15", "同上", "真实回放含 dora 记录", "PARTIAL", "重复指示倍率另在 TILE_STATE"),
    ("TABLE", "dora_indicator_red_slot_1..5", "18..22", "0/1", "dora_indicators 赤五", "red_flag", "赤五标记", "无损", "dense 表 slot16-20", "同上", "无红五指示样例", "PARTIAL", ""),
    ("TABLE", "own_rank", "23", "1..4", "scores 排序", "ranks(同分按绝对座次)", "名次", "无损", "dense 表 slot21", "同上", "真实回放抽样", "PASS", ""),
    ("TABLE", "保留列", "24..28", "必须 0", "—", "0", "—", "—", "—", "—", "semantic_validation 未校验", "PARTIAL", "校验缺口"),
    ("TABLE", "numeric scores[0..3]", "num0..3", "[-1,1]", "scores", "x/1e5 clip", "点/10万", "clip 有损", "numeric 投影×4", "同上", "域校验(numeric≤1)", "PASS", "clip 边界无单测"),
    ("TABLE", "numeric diff[0..2]", "num4..6", "[-1,1]", "scores[seat]-scores[opp]", "x/1e5 clip", "点差/10万", "clip 有损", "numeric 投影×3", "同上", "域校验", "PASS", ""),
    # ---- SELF_HAND ----
    ("SELF_HAND", "tile_type", "2", "1..34", "obs.hands[seat] 计数", "kind+1 升序", "牌种", "无损", "simple 表 slot0", "obs→Rust→共享行", "真实回放排序", "PASS", ""),
    ("SELF_HAND", "count", "3", "1..4", "hand 计数", "原值", "张数", "无损", "simple 表 slot1", "同上", "真实回放", "PASS", ""),
    ("SELF_HAND", "has_red", "4", "0/1", "hand 含赤五", "any(is_red)", "是否赤五", "无损", "simple 表 slot2", "同上", "无红五手牌断言", "PARTIAL", "缺红色断言"),
    ("SELF_HAND", "is_drawn", "5", "0/1", "drawn_tile 种类", "drawn_kind==kind", "是否当前摸牌", "无损", "simple 表 slot3", "同上", "真实回放", "PASS", ""),
    ("SELF_HAND", "locked_under_riichi", "6", "0/1", "riichi_accepted 且非当前摸牌", "契约公式", "立直锁定", "无损", "simple 表 slot4", "同上", "无立直后锁定断言", "PARTIAL", "缺测试"),
    # ---- SELF_STATE_ANALYSIS ----
    ("SELF_STATE", "menzen", "2", "0/1", "meld.opened", "all(!opened)", "门清", "无损", "dense slot0", "obs→Rust→共享行", "真实回放", "PASS", ""),
    ("SELF_STATE", "concealed_count", "3", "0..14", "obs.hands[seat].len()", "原值", "暗牌数", "无损", "dense slot1", "同上", "真实回放", "PASS", "自身暗牌数正确"),
    ("SELF_STATE", "meld_count", "4", "0..4", "obs.melds[seat].len()", "原值", "副露数", "无损", "dense slot2", "同上", "真实回放", "PASS", ""),
    ("SELF_STATE", "overall_shanten", "5", "0=和牌,1=0向听,...", "shanten::calculate", "shanten_code", "向听码", "bucket(7+ 合并)", "dense slot3", "同上", "真实回放向听未断言", "PARTIAL", "无独立向听 oracle 断言"),
    ("SELF_STATE", "standard_shanten", "6", "同上", "同", "同", "同上", "bucket", "dense slot4", "同上", "同上", "PARTIAL", ""),
    ("SELF_STATE", "chiitoitsu_shanten", "7", "0..8,9=开放N/A", "同", "127→9", "同上", "bucket+N/A", "dense slot5", "同上", "同上", "PARTIAL", "127 哨兵仅代码审查"),
    ("SELF_STATE", "kokushi_shanten", "8", "同上", "同", "同", "同上", "同上", "dense slot6", "同上", "同上", "PARTIAL", ""),
    ("SELF_STATE", "advance_kind_count", "9", "0..33,34=34+", "progress_masks", "bucket_kind_count", "进张种类", "bucket", "dense slot7", "同上", "真实回放无精确断言", "PARTIAL", ""),
    ("SELF_STATE", "advance_remaining", "10", "0..99,100=100+", "remaining 求和", "bucket_entity_count", "剩余实体", "bucket", "dense slot8", "同上", "**受被鸣牌双计影响**", "FAIL", "remaining=4-own-public 被 public 双计污染"),
    ("SELF_STATE", "wait_kind_count", "11", "0=N/A,1..33,34=34+", "progress_masks", "bucket_kind_count", "等待种类", "bucket", "dense slot9", "同上", "无精确断言", "PARTIAL", ""),
    ("SELF_STATE", "wait_remaining", "12", "0=N/A,1..99,100=100+", "remaining 求和", "bucket_entity_count", "等待剩余实体", "bucket", "dense slot10", "同上", "**受被鸣牌双计影响**", "FAIL", "同上"),
    ("SELF_STATE", "permanent_furiten", "13", "0/1", "missed_agari_riichi", "原值", "永久振听", "无损", "dense slot11", "同上", "缺失振听样例断言", "PARTIAL", ""),
    ("SELF_STATE", "doujun_furiten", "14", "0/1", "missed_agari_doujun", "原值", "同巡振听", "无损", "dense slot12", "同上", "同上", "PARTIAL", ""),
    ("SELF_STATE", "riichi_furiten", "15", "0/1", "missed_riichi && accepted", "原值", "立直振听", "无损", "dense slot13", "同上", "同上", "PARTIAL", ""),
    ("SELF_STATE", "own_dora_count", "16", "0..4,5=5+", "count_dora_aka", "bucket_dora_aka", "宝牌数", "bucket", "dense slot14", "同上", "无精确断言", "PARTIAL", ""),
    ("SELF_STATE", "own_aka_count", "17", "0..4,5=5+", "hand+meld 赤五", "min(,5)", "赤牌数", "bucket", "dense slot15", "同上", "无红五断言", "PARTIAL", ""),
    ("SELF_STATE", "own_yakuhai_han", "18", "0..5,6=6+", "open_meld_yakuhai_han", "bucket_yakuhai", "役牌番", "bucket", "dense slot16", "同上", "Rust 单测覆盖连风", "PASS", "encoding_facts 单测"),
    ("SELF_STATE", "base_han_total", "19", "0..9,10=10+", "dora+aka+yakuhai", "bucket_base_han", "基础番", "bucket", "dense slot17", "同上", "无精确断言", "PARTIAL", "名称与“确定基础番”一致(不含役)"),
    # ---- PLAYER ----
    ("PLAYER", "relative_seat", "2", "0..3", "rel_order", "相对座次", "0=自身", "无损", "dense slot0", "obs→Rust→共享行", "真实回放", "PASS", ""),
    ("PLAYER", "absolute_seat", "3", "0..3", "player", "原值", "绝对座次", "无损", "dense slot1", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "seat_wind", "4", "0..3", "seat_wind(seat,oya)", "公式", "自风", "无损", "dense slot2", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "is_oya", "5", "0/1", "player==oya", "原值", "庄家", "无损", "dense slot3", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "rank", "6", "1..4", "scores", "ranks", "名次", "无损", "dense slot4", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "concealed_count", "7", "0..14", "生命周期", "13+pending-2×chi/pon-3×daiminkan-4×ankan-1×kakan", "暗牌数", "**语义错**", "dense slot5", "同上", "**792 处与独立值不符**", "FAIL", "与契约(-3×三张,-4×杠)及真实牌数不符"),
    ("PLAYER", "meld_count", "8", "0..4", "obs.melds[p].len()", "原值", "副露数", "无损", "dense slot6", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "kan_count", "9", "0..4", "meld_type", "计数", "杠数", "无损", "dense slot7", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "menzen", "10", "0/1", "meld.opened", "all(!opened)", "门清", "无损", "dense slot8", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "river_length", "11", "0..24", "discards.len()", "min(24)", "牌河长度", "cap(24)", "dense slot9", "同上", "真实回放≤24", "PARTIAL", ""),
    ("PLAYER", "riichi_status", "12", "0/1/2", "declared/accepted", "三态", "立直状态", "无损", "dense slot10", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "riichi_turn", "13", "0=N/A,1..25,26=26+", "declaration index", "bucket_turn", "巡目", "bucket", "dense slot11", "同上", "无边界单测", "PARTIAL", ""),
    ("PLAYER", "riichi_decl_tile_type", "14", "0=N/A,1..34", "discards[declaration]", "tile_type_code", "宣言牌", "无损", "dense slot12", "同上", "真实回放", "PASS", ""),
    ("PLAYER", "riichi_decl_red", "15", "0/1", "宣言牌赤五", "red_flag", "赤五", "无损", "dense slot13", "同上", "无红五断言", "PARTIAL", ""),
    ("PLAYER", "post_riichi_discard_count", "16", "0..15,16=16+", "flags[decl+1:]", "bucket_post_riichi", "立直后舍牌数", "bucket", "dense slot14", "同上", "无边界单测", "PARTIAL", ""),
    ("PLAYER", "open_meld_yakuhai_han", "17", "0..5,6=6+", "open_meld_yakuhai_han", "bucket_yakuhai", "役牌番", "bucket", "dense slot15", "同上", "Rust 单测", "PASS", ""),
    ("PLAYER", "visible_meld_dora_aka_han", "18", "0..7,8=8+", "visible_meld_dora_aka_han", "bucket_dora_aka", "宝赤番", "bucket", "dense slot16", "同上", "Rust 单测", "PASS", ""),
    ("PLAYER", "numeric points/diff", "num0/1", "[-1,1]", "scores", "x/1e5 clip", "点/点差", "clip", "numeric 投影", "同上", "域校验", "PASS", ""),
    # ---- RIVER_SUMMARY ----
    ("RIVER_SUMMARY", "valid_length", "2", "0..6", "discards.len()", "min(6,N)", "有效槽数", "无损", "dense slot0", "obs→Rust→共享行", "真实回放逐槽独立核对", "PASS", "契约文字 i<valid_length 为文档错误(实现 i<=valid_length)"),
    ("RIVER_SUMMARY", "slot_i tile_type", "3+4(i-1)", "0=N/A,1..34", "discards", "顺序", "牌种", "无损", "dense slot(4字段求和)+slot_id", "同上", "真实回放首六/近六逐槽一致", "PASS", ""),
    ("RIVER_SUMMARY", "slot_i red", "4+4(i-1)", "0/1", "赤五", "red_flag", "赤五", "无损", "同上", "同上", "无红五摘要样例", "PARTIAL", ""),
    ("RIVER_SUMMARY", "slot_i cut", "5+4(i-1)", "0手切,1摸切,2N/A", "tsumogiri_flags", "原值", "手切/摸切", "无损", "同上", "同上", "真实回放逐槽一致", "PASS", ""),
    ("RIVER_SUMMARY", "slot_i riichi_stage", "6+4(i-1)", "0/1/2/3", "declaration index", "riichi_stage", "立直阶段", "无损", "同上", "同上", "真实回放一致", "PASS", ""),
    ("RIVER_SUMMARY", "内部槽顺序", "—", "6 槽", "—", "固定 slot_id", "位置可区分", "无损", "slot_id embedding", "同上", "顺序交换改变 embedding 测试(合成)", "PARTIAL", "槽内 4 字段为求和而非 concat，未做碰撞搜索"),
    # ---- RIVER_DISCARD ----
    ("RIVER_DISCARD", "relative_seat", "2", "1..3", "rel_order", "相对座次", "对手座次", "无损", "simple slot0", "obs→Rust→共享行", "真实回放+契约(无自身 token)", "PASS", ""),
    ("RIVER_DISCARD", "river_index", "3", "1..24", "本地序号", "index+1", "1 基序号", "无损", "simple slot1", "同上", "真实回放", "PASS", ""),
    ("RIVER_DISCARD", "tile_type", "4", "1..34", "discards", "tile_type_code", "牌种", "无损", "simple slot2", "同上", "真实回放", "PASS", ""),
    ("RIVER_DISCARD", "red", "5", "0/1", "赤五", "red_flag", "赤五", "无损", "simple slot3", "同上", "无红五断言", "PARTIAL", ""),
    ("RIVER_DISCARD", "cut", "6", "0/1", "tsumogiri_flags", "原值", "手切/摸切", "无损", "simple slot4", "同上", "真实回放", "PASS", ""),
    ("RIVER_DISCARD", "riichi_stage", "7", "0/1/2", "declaration index", "riichi_stage", "立直三态", "无损", "simple slot5", "同上", "真实回放", "PASS", ""),
    ("RIVER_DISCARD", "supplied", "8", "0/1", "meld.from_who+called_tile", "按实体牌 id 匹配", "是否被鸣", "**同牌种多张时全标**", "simple slot6", "同上", "**合成反例 (1,1)(2,1)**", "FAIL", "同牌种多次舍出仅一张被鸣时全部被标记"),
    ("RIVER_DISCARD", "age_bucket", "9", "0..3", "距最新舍牌", "distance 分桶", "年龄桶", "bucket", "simple slot7", "同上", "真实回放", "PASS", ""),
    # ---- MELD ----
    ("MELD", "owner_relative", "2", "0..3", "melds[player]", "相对座次", "owner", "无损", "dense slot0", "obs→Rust→共享行", "真实回放 pon 记录", "PASS", ""),
    ("MELD", "meld_type_code", "3", "1..5", "MeldType", "1..5", "类型", "无损", "dense slot1", "同上", "真实回放 pon；chi/杠无样例断言", "PARTIAL", ""),
    ("MELD", "tile0..3_type/red", "4..11", "type 1..34/red 0/1", "meld.tiles", "顺序", "构成牌", "无损", "dense slot2-9", "同上", "真实回放 pon 记录", "PARTIAL", "缺 chi/kan/kakan 样例"),
    ("MELD", "called_tile_type/red", "12/13", "0=N/A,1..34", "meld.called_tile", "tile_type_code", "被鸣牌", "无损", "dense slot10-11", "同上", "真实回放 pon 记录", "PASS", ""),
    ("MELD", "supplier_relative", "14", "0=N/A,1..3", "meld.from_who", "relative_code", "供牌者", "无损(依赖 from_who)", "dense slot12", "同上", "真实回放 from_who 正确", "PASS", "apply_mjai_event 路径 from_who=-1 会导致 0"),
    ("MELD", "open", "15", "0/1", "meld.opened", "原值", "开放", "无损", "dense slot13", "同上", "真实回放", "PASS", ""),
    ("MELD", "meld_index", "16", "1..4", "枚举", "index+1", "副露序号", "无损", "dense slot14", "同上", "真实回放", "PASS", ""),
    ("MELD", "yakuhai_han", "17", "0..5,6=6+", "open_meld_yakuhai_han", "bucket", "役牌番", "bucket", "dense slot15", "同上", "Rust 单测", "PASS", ""),
    ("MELD", "visible_dora_aka_han", "18", "0..7,8=8+", "visible_meld_dora_aka_han", "bucket", "宝赤番", "bucket", "dense slot16", "同上", "Rust 单测", "PASS", ""),
    # ---- TILE_STATE ----
    ("TILE_STATE", "tile_type", "2", "1..34", "—", "升序 1..34", "牌种", "无损", "simple slot0", "obs→Rust→共享行", "真实回放+结构校验", "PASS", ""),
    ("TILE_STATE", "self_concealed_count", "3", "0..4", "own hand", "tile_counts", "暗手数", "无损", "simple slot1", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "self_discard_count", "4", "0..4", "own river", "tile_counts", "自己舍牌数", "无损", "simple slot2", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "self_ever_discarded", "5", "0/1", "own river", ">0", "是否舍过", "无损", "simple slot3", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "public_count", "6", "0..4", "四家牌河+副露+指示", "区域计数求和 cap4", "公开张数", "**被鸣牌双计**", "simple slot4", "同上", "**703 处 public>实体数**", "FAIL", "被鸣牌同时在 river 与 meld 中重复计数"),
    ("TILE_STATE", "known_count", "7", "0..4", "public+own", "min(4,sum)", "已知张数", "**受双计影响**", "simple slot5", "同上", "**688 处 known 高于实体已知**", "FAIL", "同上"),
    ("TILE_STATE", "unknown_count", "8", "0..4", "4-known", "原值", "未知实体", "**受双计影响**", "simple slot6", "同上", "同上", "FAIL", "同上"),
    ("TILE_STATE", "all_seen", "9", "0/1", "unknown==0", "原值", "全见", "**受双计影响**", "simple slot7", "同上", "同上", "FAIL", "同上"),
    ("TILE_STATE", "dora_multiplicity", "10", "0..5", "dora_indicators", "saturating_add", "倍率", "无损(≤5)", "simple slot8", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "is_dora", "11", "0/1", "multiplicity>0", "原值", "是否宝牌", "无损", "simple slot9", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "round_wind_match", "12", "0/1", "round_wind", "kind==wind", "场风", "无损", "simple slot10", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "seat_wind_match", "13", "0/1", "seat_wind", "kind==wind", "自风", "无损", "simple slot11", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "red_five_kind", "14", "0/1", "kind∈{4,13,22}", "原值", "赤五牌种", "无损", "simple slot12", "同上", "无红五断言", "PARTIAL", ""),
    ("TILE_STATE", "is_advance", "15", "0/1", "progress_masks", "掩码", "进张", "无损", "simple slot13", "同上", "无精确断言", "PARTIAL", ""),
    ("TILE_STATE", "is_win", "16", "0/1", "progress_masks", "掩码", "和牌", "无损", "simple slot14", "同上", "无精确断言", "PARTIAL", ""),
    ("TILE_STATE", "genbutsu_shimo/toimen/kamicha", "17..19", "0/1", "对手牌河", "river_mask", "现物", "无损", "simple slot15-17", "同上", "真实回放", "PASS", ""),
    ("TILE_STATE", "suji_shimo/toimen/kamicha", "20..22", "0/1/2/3", "对手牌河", "suji_category", "筋类别", "无损", "simple slot18-20", "同上", "无筋边界断言", "PARTIAL", ""),
    ("TILE_STATE", "wall_class", "23", "0/1/2", "analysis::wall_class", "相关牌可见数≥3/4", "壁", "等价类", "simple slot21", "同上", "Rust 单测分析层", "PARTIAL", ""),
    ("TILE_STATE", "dora_neighbor", "24", "0/1", "dora 邻张", "同花色差1", "宝牌邻张", "无损", "simple slot22", "同上", "无精确断言", "PARTIAL", ""),
    # ---- OPPONENT_ANALYSIS ----
    ("ANALYSIS", "relative_seat", "2", "1..3", "rel_order", "相对座次", "对手", "无损", "dense slot0", "obs→Rust→analysis 行", "真实回放", "PASS", ""),
    ("ANALYSIS", "riichi_status/turn/decl", "3..6", "同 PLAYER", "同 PLAYER", "同", "同", "同", "dense slot1-4", "同上", "真实回放", "PASS", ""),
    ("ANALYSIS", "menzen", "7", "0/1", "meld.opened", "all(!opened)", "门清", "无损", "dense slot5", "同上", "真实回放", "PASS", ""),
    ("ANALYSIS", "concealed_count", "8", "0..14", "生命周期", "同 PLAYER(错误公式)", "暗牌数", "**语义错**", "dense slot6", "同上", "**与 PLAYER 同缺陷**", "FAIL", ""),
    ("ANALYSIS", "meld_count/kan_count", "9/10", "0..4", "obs.melds", "计数", "副露/杠数", "无损", "dense slot7-8", "同上", "真实回放", "PASS", ""),
    ("ANALYSIS", "open_meld_yakuhai_han", "11", "0..5,6=6+", "同 PLAYER", "bucket", "役牌番", "bucket", "dense slot9", "同上", "Rust 单测", "PASS", ""),
    ("ANALYSIS", "visible_meld_dora_aka_han", "12", "0..7,8=8+", "同 PLAYER", "bucket", "宝赤番", "bucket", "dense slot10", "同上", "Rust 单测", "PASS", ""),
    ("ANALYSIS", "post_riichi_tedashi/tsumogiri", "13/14", "0..15,16=16+", "flags[decl+1:]", "bucket_post_riichi", "立直后切/摸切", "bucket", "dense slot11-12", "同上", "无边界单测", "PARTIAL", ""),
    ("ANALYSIS", "recent6_tedashi/tsumogiri", "15/16", "0..6", "flags[-6:]", "min(6)", "近六切/摸切", "bucket(clamp6)", "dense slot13-14", "同上", "真实回放", "PARTIAL", ""),
    ("ANALYSIS", "own_genbutsu_kind_count", "17", "0..33,34=34+", "自身手 vs 对手河", "bucket_kind_count", "现物种类", "bucket", "dense slot15", "同上", "无精确断言", "PARTIAL", ""),
    ("ANALYSIS", "own_genbutsu_entity_count", "18", "0..99,100=100+", "自身手 vs 对手河", "bucket_entity_count", "现物实体", "bucket", "dense slot16", "同上", "无精确断言", "PARTIAL", ""),
    ("ANALYSIS", "river_length", "19", "0..24", "discards.len()", "min(24)", "牌河长度", "cap", "dense slot17", "同上", "真实回放", "PASS", ""),
    # ---- ACTION QUERY ----
    ("ACTION", "action_type_code", "2", "1..11", "action 类型", "映射表", "类型", "无损", "dense slot0", "obs→query Rust→actor 行", "真实回放+语义校验", "PASS", ""),
    ("ACTION", "primary_tile_code", "3", "0=N/A,1..34", "action.tile", "kind+1", "主牌", "无损", "dense slot1", "同上", "真实回放", "PASS", ""),
    ("ACTION", "source_seat_code", "4", "0=N/A,1..3", "last_offer_actor", "relative", "供牌者", "无损", "dense slot2", "同上", "supplier 域校验", "PASS", ""),
    ("ACTION", "tsumogiri_mode", "5", "0/1", "action_id 奇偶恢复", "1..74 内 (id-1)%2", "是否摸切", "无损(依赖 241 映射)", "dense slot3", "同上", "真实回放", "PARTIAL", "由 action_id 数学推断，非动作真实字段"),
    ("ACTION", "answer O0..O9", "6..15", "各自基数", "Rust query 内核", "bucket 映射", "动作后分析", "bucket", "dense slot4-13", "同上", "域校验(上限自证)", "PARTIAL", "无独立语义值 oracle"),
    ("ACTION", "answer D0..D9", "6..15", "各自基数", "Rust query 内核", "bucket 映射", "防守分析", "bucket", "dense slot4-13", "同上", "同上", "PARTIAL", ""),
    ("ACTION", "action_id / consume 组合", "—(不在 token)", "0..240", "action_id / consume_tiles", "仅 scatter/query_rows", "—", "**token 中丢失**", "logits scatter", "query_rows 仅离线校验", "**12 个真实决策 O/D 特征完全相同**", "FAIL", "违反 FR-14：consume 组合/action_id 未进入 embedding"),
    # ---- CRITIC ----
    ("CRITIC_HAND", "relative_seat/tile_type/red/count", "2..5", "1..3/1..34/0/1..4", "priv hands(离线桥)", "原值", "闭手", "无损", "simple tables", "bridge critic_features", "单元测试覆盖红五/顺序", "PASS", "仅合成；真实离线桥未单测"),
    ("CRITIC_FUTURE", "position/tile_type/red", "2..4", "1..5/1..34/0/1", "未来五张(离线桥)", "原值", "未来牌", "无损", "simple tables", "同上", "单元测试覆盖", "PASS", "walls=None 时未来五张缺失"),
]

HEADER = "| kind | 字段 | 偏移 | 域 | 真值来源 | 编码公式 | 独立解码 | 有损性 | 模型消费 | 数据链 | 自动化证据 | 判定 | 说明 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"


def main() -> int:
    lines = [
        "# V18 当前局面输入字段级审查矩阵",
        "",
        "由 `audit/reports/v18/scripts/gen_field_matrix.py` 生成；字段定义取自",
        "`specs/010-v18-current-state-input-sft/contracts/v18-current-state-contract.md` §3",
        "（审查基准，不读生产 schema）。判定列：PASS / PARTIAL / FAIL / UNTESTED。",
        "",
        HEADER,
    ]
    for row in ROWS:
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "## 判定统计",
        "",
        f"- PASS：{sum(1 for r in ROWS if r[11] == 'PASS')}",
        f"- PARTIAL：{sum(1 for r in ROWS if r[11] == 'PARTIAL')}",
        f"- FAIL：{sum(1 for r in ROWS if r[11] == 'FAIL')}",
        f"- UNTESTED：{sum(1 for r in ROWS if r[11] == 'UNTESTED')}",
        "",
        "## FAIL 汇总",
        "",
    ]
    for row in ROWS:
        if row[11] == "FAIL":
            lines.append(f"- **{row[0]} / {row[1]}**（{row[3]}）：{row[12]}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {OUT} ({len(ROWS)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
