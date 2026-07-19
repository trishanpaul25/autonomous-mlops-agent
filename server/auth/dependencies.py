from uuid import UUID

from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from server.auth.jwt import verify_access_token
from server.db.session import get_db
from server.repositories.user_repository import UserRepository
from server.models.user import User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login"
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:

    payload = verify_access_token(token)

    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials.",
        )

    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials.",
        )

    repository = UserRepository(db)

    user = repository.get_by_id(
        UUID(user_id),
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found.",
        )

    return user