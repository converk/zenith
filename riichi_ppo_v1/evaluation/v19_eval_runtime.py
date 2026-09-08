"""2v2 跨代评测的 V19 侧运行时引导(仅 stdlib,先于模型导入链执行)。

锁步要求两侧环境/状态机对同一物理局面产生逐字节相同的事件流(含 dahai 的
tsumogiri 标记)。本模块把 ``--v19-runtime-dir`` 指定的 V19 扩展新鲜构建
(与 V18 工作副本构建同源同特征,行为已验证一致)预注册进 ``sys.modules``,
使 host 进程的 ``import riichi``/``import riichienv`` 落到该构建上;不修改
站点已安装扩展,不影响任何其他进程。仅供评测入口使用,训练路径禁止调用。
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def ensure_v19_runtime(runtime_dir: str | Path) -> None:
    """预注册 V19 评测运行时扩展;须先于任何 riichi/riichienv 导入调用。"""
    if "riichi" in sys.modules or "riichienv" in sys.modules:
        raise RuntimeError("V19 评测运行时必须在导入 riichi/riichienv 之前引导")
    runtime_dir = Path(runtime_dir).resolve()
    riichi_so = runtime_dir / "libriichi.so"
    native_so = runtime_dir / "lib_riichienv.so"
    for so_path in (riichi_so, native_so):
        if not so_path.is_file():
            raise RuntimeError(
                f"V19 评测运行时扩展缺失: {so_path};请用当前源码构建 "
                "`cargo build --release -p riichienv-state-machine "
                "-p riichienv-python --features pyo3/extension-module` "
                "并提供其产物目录"
            )

    def load_extension(so_path: Path, name: str):
        loader = importlib.machinery.ExtensionFileLoader(name, str(so_path))
        spec = importlib.util.spec_from_loader(name, loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)
        return module

    load_extension(riichi_so, "riichi")
    native = load_extension(native_so, "_riichienv")
    sys.modules.pop("_riichienv", None)
    env_pkg_dir = (
        Path(__file__).resolve().parent.parent.parent / "RiichiEnv" / "src" / "riichienv"
    )
    sys.modules["riichienv._riichienv"] = native
    spec = importlib.util.spec_from_file_location(
        "riichienv",
        env_pkg_dir / "__init__.py",
        submodule_search_locations=[str(env_pkg_dir)],
    )
    env_pkg = importlib.util.module_from_spec(spec)
    sys.modules["riichienv"] = env_pkg
    spec.loader.exec_module(env_pkg)
