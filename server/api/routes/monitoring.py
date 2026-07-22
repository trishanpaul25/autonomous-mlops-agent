from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.db.session import get_db
from server.schemas import MonitoringResponse
from server.services.monitoring_service import MonitoringService
from server.repositories.monitoring_repository import MonitoringRepository

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)


@router.post(
    "/{deployment_id}/check",
    response_model=MonitoringResponse,
)
def run_monitoring_check(deployment_id: UUID, db: Session = Depends(get_db)):
    """
    Triggers a fresh monitoring check for a deployment: aggregates all
    PredictionLog rows recorded for it against its DatasetSnapshot,
    and persists a new Monitoring row with the result.
    """
    try:
        return MonitoringService(db).run_check(deployment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get(
    "/{deployment_id}",
    response_model=MonitoringResponse,
)
def get_latest_monitoring(deployment_id: UUID, db: Session = Depends(get_db)):
    """
    Returns the most recent monitoring check for a deployment, without
    triggering a new one.
    """
    result = MonitoringRepository(db).get_latest_by_deployment_id(deployment_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No monitoring checks found for deployment '{deployment_id}'.",
        )
    return result
