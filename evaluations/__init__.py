"""麻将策略模型评测脚本包。

功能：
    保存所有和 checkpoint 评测相关的脚本，包括模型对局评测、Elo 排名统计，
    以及根据 checkpoint 目录自动触发的定时评测。
    evaluations/config.py 是默认配置文件；evaluations/core/ 是内部实现模块；
    外层其他文件主要是命令行入口。

使用方法：
    编辑 evaluations/config.py
    python -m evaluations.checkpoint_match
    python -m evaluations.elo_ranking
    python -m evaluations.scheduled_checkpoint_match
"""
