"""
Base state shared by all agent states.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseState(BaseModel):
    """
    Base class for every state in the pipeline.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    state_id: str = Field(default_factory=lambda: str(uuid4()))

    created_at: datetime = Field(default_factory=datetime.utcnow)

    updated_at: datetime = Field(default_factory=datetime.utcnow)

    status: str = "pending"

    error: str | None = None