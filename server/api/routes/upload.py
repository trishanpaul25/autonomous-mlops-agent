from datetime import datetime
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile

from sqlalchemy.orm import Session

from server.core.config import settings
from server.db.session import get_db
from server.models.dataset import Dataset
from server.repositories.dataset_repository import DatasetRepository
from server.schemas import UploadResponse
from server.services.upload_service import upload_service
from server.models.user import User
from server.auth.dependencies import get_current_user

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)


@router.post("/", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    extension = "." + file.filename.split(".")[-1].lower()

    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    dataset_info = await upload_service.save_file(file)

    dataset = Dataset(
        id=UUID(dataset_info["dataset_id"]),
        user_id=None,        # Authentication will be added later
        dataset_name=dataset_info["dataset_name"],
        filename=dataset_info["filename"],
        dataset_path=dataset_info["dataset_path"],
        source_type=dataset_info["source_type"],
        file_size=dataset_info["file_size"],
        uploaded_at=datetime.utcnow(),
    )

    repository = DatasetRepository(db)

    repository.create(dataset)
    
    return UploadResponse(**dataset_info)