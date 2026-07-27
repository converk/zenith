"""CQL trainer — re-exports bc_logs.Trainer with CQL-focused defaults.

The CQL training logic is identical to BC with alpha > 0,
so this module simply re-exports the shared Trainer class.
"""
from riichienv_ml.trainers.bc_logs import Trainer, cql_loss

__all__ = ["Trainer", "cql_loss"]
