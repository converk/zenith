"""RiichiEnv 的牌维与编码常量。

领域不变常量在此单一命名定义:136 张实体牌 TID、34 类牌、三麻 27 类牌。
"""

TID_COUNT = 136  # 34 类牌 × 4 张实体牌
TILE_KINDS = 34  # 万/饼/索各 9 类 + 风/三元 7 类
N_TILE_TYPES_3P = 27  # 1m, 9m, 1-9p, 1-9s, 4 winds, 3 dragons (no 2m-8m)
