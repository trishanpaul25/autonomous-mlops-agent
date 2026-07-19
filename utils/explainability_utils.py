"""
explainability_utils.py

Small, dependency-light helpers shared by explainability_tool.py and
explainability_agent.py. Kept separate so neither the pure-computation
layer nor the orchestration layer has to re-implement serialization or
logging setup.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List


def get_agent_logger(name: str = "explainability_agent") -> logging.Logger:
    """Return a module-level logger with a consistent, structured format.
    Idempotent: safe to call repeatedly without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays into native
    Python types so results are safe to store on PipelineState and
    serialize to JSON. Avoids a hard numpy/pandas import requirement by
    duck-typing on the presence of `.tolist()` / `.item()`.
    """
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict, str)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)):
        return value
    # Last resort: best-effort string conversion so we never raise while
    # packaging results.
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def normalize_to_unit_range(values: Iterable[float]) -> List[float]:
    """Min-max normalize a sequence of non-negative magnitudes to [0, 1].
    Used to make SHAP / permutation / native / coefficient importances
    comparable when building the unified ranking. Returns zeros if the
    input has no spread.
    """
    values = list(values)
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)