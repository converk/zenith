from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

import riichi

import riichienv


def test_riichi_module_remains_public() -> None:
    assert hasattr(riichi, "MjaiKyokuStateMachineManager")
    assert riichi.ANALYSIS_VERSION == 4


def test_riichi_does_not_import_riichienv() -> None:
    code = textwrap.dedent(
        """
        import sys
        import riichi
        assert "riichienv" not in sys.modules, sorted(
            name for name in sys.modules if "riichienv" in name
        )
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_riichienv_no_longer_lazy_loads_visualizer() -> None:
    source = inspect.getsource(riichienv)
    assert "riichienv.visualizer" not in source
    assert not hasattr(riichienv.RiichiEnv, "get_viewer")
