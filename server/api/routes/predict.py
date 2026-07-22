import time
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.schemas import PredictRequest, PredictResponse
from server.db.session import get_db
from server.repositories.deployment_repository import DeploymentRepository
from server.repositories.prediction_log_repository import PredictionLogRepository
from server.models.prediction_log import PredictionLog
from tools.deployment.model_server_registry import ModelServerRegistry
from utils.logger import logger

router = APIRouter(
    prefix="/predict",
    tags=["Deployment"],
)


@router.post(
    "/{deployment_id}",
    response_model=PredictResponse,
)
def predict(deployment_id: str, request: PredictRequest, db: Session = Depends(get_db)):
    """
    Score raw input rows against a model the Deployment Agent has
    already loaded into the in-process ModelServerRegistry for this
    pipeline run. `deployment_id` is the pipeline run_id.
    """
    model = ModelServerRegistry.get(deployment_id)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No deployed model found for deployment_id '{deployment_id}'. "
                "It may not have been deployed yet, or the server has "
                "restarted since it was."
            ),
        )

    try:
        input_df = pd.DataFrame(request.records)
        start = time.perf_counter()
        result_df = model.predict(input_df)
        latency_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        logger.error("[Predict] Inference failed for '%s': %s", deployment_id, exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}")

    predictions = result_df.to_dict(orient="records")

    # Log the call for monitoring. This must never affect the response
    # that's already been correctly computed above — a logging failure
    # (bad run_id format, no matching deployment row yet, DB down) just
    # means this call won't be visible to the Monitoring agent, not
    # that the caller's request fails.
    try:
        deployment_record = DeploymentRepository(db).get_by_run_id(UUID(deployment_id))
        if deployment_record is not None:
            PredictionLogRepository(db).create(
                PredictionLog(
                    deployment_id=deployment_record.id,
                    input_payload=request.records,
                    prediction=predictions,
                    latency_ms=latency_ms,
                )
            )
        else:
            logger.warning(
                "[Predict] No deployment row found for run_id '%s' — "
                "prediction served but not logged for monitoring.",
                deployment_id,
            )
    except Exception as exc:
        logger.warning(
            "[Predict] Failed to write prediction log for '%s': %s",
            deployment_id,
            exc,
        )

    return PredictResponse(
        deployment_id=deployment_id,
        predictions=predictions,
    )

