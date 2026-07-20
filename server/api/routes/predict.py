import pandas as pd
from fastapi import APIRouter, HTTPException

from server.schemas import PredictRequest, PredictResponse
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
def predict(deployment_id: str, request: PredictRequest):
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
        result_df = model.predict(input_df)
    except Exception as exc:
        logger.error("[Predict] Inference failed for '%s': %s", deployment_id, exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}")

    return PredictResponse(
        deployment_id=deployment_id,
        predictions=result_df.to_dict(orient="records"),
    )
