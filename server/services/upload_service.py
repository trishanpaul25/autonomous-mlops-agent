"""
Service responsible for handling dataset uploads.
"""

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from server.core.paths import UPLOAD_DIR
from server.services.dataset_registry import dataset_registry

class UploadService:
    """
    Handles dataset upload and storage.
    """

    async def save_file(self, file: UploadFile) -> dict:
        # Preserve original extension
        extension = Path(file.filename).suffix

        # Unique filename
        dataset_id = str(uuid4())

        filename = f"{dataset_id}{extension}"

        destination = UPLOAD_DIR / filename

        #dataset_registry.add_dataset(
        # dataset_id=dataset_id,
        #  path=str(destination),
        #  filename=file.filename
        #)

        contents = await file.read()

        with open(destination, "wb") as f:
            f.write(contents)

        extension = Path(file.filename).suffix.lower()

        source_map = {
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".zip": "zip",
        }

        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "dataset_name": Path(file.filename).stem,
            "dataset_path": str(destination),
            "source_type": source_map.get(extension),
            "file_size": len(contents),
        }


upload_service = UploadService()