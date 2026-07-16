from pydantic import BaseModel


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    dataset_name: str
    dataset_path: str
    source_type: str
    file_size: int