"""Frozen schema-11 inference support.

This package must not be imported by v13 training, preprocessing, or loading.
"""

from .adapter import V11PolicyAdapter
from .contract import V11_CONTRACT_ID

__all__ = ["V11PolicyAdapter", "V11_CONTRACT_ID"]
