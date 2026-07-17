from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import BigInteger
from sqlalchemy import TIMESTAMP
from sqlalchemy import ForeignKey
from sqlalchemy import Boolean

from sqlalchemy.dialects.postgresql import UUID

from server.db.database import Base


class Dataset(Base):

    __tablename__ = "datasets"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
    )

    dataset_name = Column(
        String,
        nullable=False,
    )

    filename = Column(
        String,
        nullable=False,
    )

    dataset_path = Column(
        Text,
        nullable=False,
    )

    source_type = Column(
        String,
        nullable=False,
    )

    file_size = Column(
        BigInteger,
    )

    uploaded_at = Column(
        TIMESTAMP,
    )

    is_deleted = Column(
        Boolean,
        nullable=False,
        default=False,
    )