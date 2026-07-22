from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class UserProfileResponse(BaseModel):
    id: UUID
    username: str
    email: str
    created_at: datetime

    total_runs: int
    successful_runs: int
    failed_runs: int

    class Config:
        from_attributes = True