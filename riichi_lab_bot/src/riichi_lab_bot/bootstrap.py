"""Import-light console bootstrap that installs CUDA visibility first."""

from __future__ import annotations

import os


def main() -> None:
    cuda_device = os.environ.get("CUDA_DEVICE")
    if cuda_device and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    from .cli import main as cli_main

    cli_main()

