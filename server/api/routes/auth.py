from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session

from server.db.session import get_db

from server.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
)

from server.services.auth_service import AuthService

from server.auth.dependencies import get_current_user

from server.models.user import User


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

@router.post(
    "/register",
    response_model=UserResponse,
)
def register(
    request: RegisterRequest,
    db: Session = Depends(get_db),
):
    auth_service = AuthService(db)

    try:
        user = auth_service.register(
            username=request.username,
            email=request.email,
            password=request.password,
        )

        return user

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db),
):
    auth_service = AuthService(db)

    try:
        token = auth_service.login(
            request.email,
            request.password,
        )

        return TokenResponse(
            access_token=token,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail=str(exc),
        )
    


@router.get(
    "/me",
    response_model=UserResponse,
)
def me(
    current_user: User = Depends(
        get_current_user,
    ),
):
    return current_user