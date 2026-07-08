"""评测内部实现模块。

功能：
    收纳不会直接从命令行调用的评测实现代码，包括对局执行、Elo 排名
    计算、定时任务调度等。外层 evaluations/*.py 文件主要作为入口脚本。

使用方法：
    通常不需要直接运行这里的文件；请使用：
    python -m evaluations.scheduled_checkpoint_match
"""
