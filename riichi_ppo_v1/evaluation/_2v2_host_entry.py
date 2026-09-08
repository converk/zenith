"""2v2 跨代评测 host 进程入口。

先解析并执行 V19 评测运行时引导(必须先于本包内任何 ``import riichienv``
的模块级导入),再把其余参数原样委托给 ``head_to_head_2v2.main``。
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    runtime_dir: str | None = None
    rest: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == "--v19-runtime-dir":
            runtime_dir = args[index + 1]
            index += 2
        elif args[index].startswith("--v19-runtime-dir="):
            runtime_dir = args[index].split("=", 1)[1]
            index += 1
        else:
            rest.append(args[index])
            index += 1
    if not runtime_dir:
        raise SystemExit("--v19-runtime-dir is required for the 2v2 host entry")
    from .v19_eval_runtime import ensure_v19_runtime

    ensure_v19_runtime(runtime_dir)
    sys.argv = [sys.argv[0], *rest]
    from .head_to_head_2v2 import main as host_main

    host_main()


if __name__ == "__main__":
    main()
