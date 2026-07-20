from datetime import datetime

from fastapi import UploadFile
from sqlalchemy.orm import Session

from server.models.dataset import Dataset
from server.models.user import User
from server.repositories.dataset_repository import DatasetRepository
from server.services.upload_service import upload_service


class DatasetService:

    def __init__(self, db: Session):
        self.repository = DatasetRepository(db)

    def create_uploaded_dataset(
        self,
        user: User,
        upload_info: dict,
    ) -> Dataset:
        
        dataset = Dataset(
          id=upload_info["id"],
          user_id=user.id,
          dataset_name=upload_info["dataset_name"],
          filename=upload_info["filename"],
          dataset_path=upload_info["dataset_path"],
          source_type=upload_info["source_type"],
          file_size=upload_info["file_size"],
          uploaded_at=datetime.utcnow(),
        )

        return self.repository.create(dataset)
    
