"""Plugin discovery for third-party evaluators (e.g. Mortal, akochan, kanachan).

Evaluator plugins register via the ``riichienv_ml.evaluators`` entry-point
group.  The main training code calls :func:`load_evaluator` which returns
``None`` when the plugin is not installed, enabling graceful degradation.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Literal, Protocol, runtime_checkable

from loguru import logger

ENTRY_POINT_GROUP = "riichienv_ml.evaluators"


@runtime_checkable
class ThirdPartyEvaluator(Protocol):
    """Minimal interface that evaluator plugins must satisfy."""

    def evaluate(self, hero_weights: dict, num_episodes: int = 48) -> dict: ...
    def metrics_to_logline(self, metrics: dict) -> str: ...


def load_evaluator(
    evaluator_name: Literal["mortal", "riichienv", "akochan", "kanachan"] = "mortal",
    **kwargs,
) -> ThirdPartyEvaluator | None:
    """Discover and instantiate an evaluator plugin by name.

    Returns ``None`` if the plugin is not installed or fails to initialise.
    """
    eps = entry_points(group=ENTRY_POINT_GROUP)
    matched = [ep for ep in eps if ep.name == evaluator_name]
    if not matched:
        logger.info(
            f"No evaluator plugin '{evaluator_name}' found "
            f"(group={ENTRY_POINT_GROUP}). Skipping."
        )
        return None

    ep = matched[0]
    try:
        cls = ep.load()
        evaluator = cls(**kwargs)
        logger.info(f"Loaded evaluator plugin '{evaluator_name}' via entry_points")
        return evaluator
    except Exception as e:
        logger.warning(f"Failed to load evaluator plugin '{evaluator_name}': {e}")
        return None
