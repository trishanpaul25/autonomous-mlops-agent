from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from server.db.session import get_db

from server.schemas import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    UserProfileResponse,
    UpdateUserRequest,
    ChangePasswordRequest,
)
from server.repositories.pipeline_run_repository import PipelineRunRepository

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
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    auth_service = AuthService(db)

    try:
        token = auth_service.login(
            form_data.username,
            form_data.password,
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
    response_model=UserProfileResponse,
)
def me(
    current_user: User = Depends(
        get_current_user,
    ),
    db: Session = Depends(get_db),
):
    repository = PipelineRunRepository(db)

    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,

        total_runs=repository.count_by_user(current_user.id),
        successful_runs=repository.count_successful(current_user.id),
        failed_runs=repository.count_failed(current_user.id),
    )

@router.put(
    "/me",
    response_model=UserResponse,
)
def update_profile(
    request: UpdateUserRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    auth_service = AuthService(db)

    try:
        return auth_service.update_profile(
            user=current_user,
            username=request.username,
            email=request.email,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    

@router.put(
    "/change-password",
)
def change_password(
    request: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    auth_service = AuthService(db)

    try:
        auth_service.change_password(
            user=current_user,
            current_password=request.current_password,
            new_password=request.new_password,
        )

        return {
            "message": "Password updated successfully."
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    
@router.delete("/me")
def delete_account(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    auth_service = AuthService(db)

    auth_service.deactivate_account(current_user)

    return {
        "message": "Account deactivated successfully."
    }