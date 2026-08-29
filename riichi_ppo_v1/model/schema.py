"""PPO 与 SFT 共用的版本化序列化与领域常量。

领域不变常量(动作空间维度、牌类数)在此单一命名定义,其余模块一律引用,禁止
散落魔法数字。序列化协议版本引用 `encoding_protocol.py` 的单一协议版本常量。
"""

from .encoding_protocol import ENCODING_PROTOCOL_VERSION, NUM_ACTIONS

# 与现行信息编码协议保持一致(单一版本,单一来源)。
TOKEN_SCHEMA_VERSION = ENCODING_PROTOCOL_VERSION
# 固定 241 维动作空间(协议维度,不随实验版本变化;单源为 encoding_protocol.NUM_ACTIONS)。
# 34 类牌(万/饼/索各 9 类 + 风/三元 7 类)。
TILE_KINDS = 34
# 136 张实体牌 TID(34 类牌 × 4 张)。
TID_COUNT = 136
# 四人麻将的玩家数(环境观察者/座位数,领域不变常量)。
NUM_PLAYERS = 4

# 包内领域常量的公开面(NUM_ACTIONS 为 encoding_protocol 的显式再导出)。
__all__ = ["ENCODING_PROTOCOL_VERSION", "NUM_ACTIONS", "NUM_PLAYERS", "TID_COUNT", "TILE_KINDS", "TOKEN_SCHEMA_VERSION"]
