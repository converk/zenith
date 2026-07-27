"""Plugin discovery for teacher workers.

Teacher plugins register via the ``riichienv_ml.teachers`` entry-point group.
The online BC trainer calls :func:`load_teacher_worker_class` to discover a
Ray remote worker class that generates training data by playing games.
"""
from __future__ import annotations

from importlib.metadata import entry_points

from loguru import logger

ENTRY_POINT_GROUP = "riichienv_ml.teachers"


ENTRY_POINT_NAME = "teacher_worker"


def load_teacher_worker_class():
    """Discover the teacher worker class (Ray remote) via entry_points.

    Returns the class itself (not an instance).
    Raises RuntimeError if the plugin is not installed.
    """
    eps = entry_points(group=ENTRY_POINT_GROUP)
    matched = [ep for ep in eps if ep.name == ENTRY_POINT_NAME]
    if not matched:
        raise RuntimeError(
            f"No teacher plugin '{ENTRY_POINT_NAME}' found "
            f"(group={ENTRY_POINT_GROUP}). "
            f"Install riichienv-ml-xt to enable teacher workers."
        )

    cls = matched[0].load()
    logger.info(f"Loaded teacher plugin '{ENTRY_POINT_NAME}' via entry_points")
    return cls
