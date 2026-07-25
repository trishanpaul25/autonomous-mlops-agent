from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from server.auth.dependencies import get_current_user
from server.db.session import get_db
from server.models.user import User
from server.schemas import DeploymentSummary, DeploymentDetail
from server.services.deployment_service import DeploymentService

router = APIRouter(
    prefix="/deployments",
    tags=["Deployments"],
)


@router.get(
    "/",
    response_model=list[DeploymentSummary],
)
def list_deployments(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Every deployment the current user has created, newest first.
    Powers the Deployments page: model name, live/cold status,
    metrics, and created time in one call.
    """
    return DeploymentService(db).list_for_user(current_user.id)


@router.get(
    "/{deployment_id}",
    response_model=DeploymentDetail,
)
def get_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    `deployment_id` is the pipeline run_id — the same value used in
    POST /predict/{deployment_id} — not the internal `deployments.id`.
    Includes the latest monitoring snapshot, if one has been run.
    """
    try:
        return DeploymentService(db).get_detail(deployment_id, current_user.id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
