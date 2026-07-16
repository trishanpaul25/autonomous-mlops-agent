from fastapi import APIRouter
from fastapi import File
from fastapi import UploadFile
from fastapi import HTTPException

from server.schemas import UploadResponse
from server.services.upload_service import upload_service
from server.services.dataset_registry import dataset_registry
from server.core.config import settings

router = APIRouter(
    prefix="/upload",
    tags=["Upload"]
)


@router.post("/", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...)
):
    extension = "." + file.filename.split(".")[-1].lower()

    if extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )

    dataset_info = await upload_service.save_file(file)

    dataset_registry.add_dataset(dataset_info)

    return UploadResponse(**dataset_info)