from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from uuid import uuid4
from server.db.database import Base
from sqlalchemy.sql import func
from sqlalchemy import Boolean
class User(Base):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True),
        default=uuid4,
        primary_key=True,
    )

    username = Column(
        String,
        nullable=False,
    )

    email = Column(
        String,
        unique=True,
        nullable=False,
    )

    password_hash = Column(
        String,
        nullable=False,
    )

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
)