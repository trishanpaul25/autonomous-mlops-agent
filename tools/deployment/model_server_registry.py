"""
Model Server Registry.

Process-wide, in-memory cache of loaded MLflow pyfunc models, keyed by
deployment_id. This is what makes "deployment" real for a locally
served FastAPI route: DeploymentAgent populates it once a pipeline run
completes, and server/api/routes/predict.py reads from it on every
/predict request — no reload from disk per request.

Deliberately process-local (a plain dict), not persisted or shared
across workers/restarts. If the server restarts, deployments are gone
until the next pipeline run re-populates them — an accepted limitation
of "local FastAPI route" serving, not a bug.
"""

from __future__ import annotations

import threading
from typing import Any


class ModelServerRegistry:
    """Thread-safe singleton cache of loaded pyfunc models."""

    _lock = threading.Lock()
    _models: dict[str, Any] = {}

    @classmethod
    def register(cls, deployment_id: str, model: Any) -> None:
        with cls._lock:
            cls._models[deployment_id] = model

    @classmethod
    def get(cls, deployment_id: str) -> Any | None:
        with cls._lock:
            return cls._models.get(deployment_id)

    @classmethod
    def unregister(cls, deployment_id: str) -> None:
        with cls._lock:
            cls._models.pop(deployment_id, None)

    @classmethod
    def is_registered(cls, deployment_id: str) -> bool:
        with cls._lock:
            return deployment_id in cls._models
