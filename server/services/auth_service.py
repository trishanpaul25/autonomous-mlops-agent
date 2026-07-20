from sqlalchemy.orm import Session
from uuid import UUID
from server.models.user import User
from server.repositories.user_repository import UserRepository
from server.core.password import (
    hash_password,
    verify_password,
)
from server.auth.jwt import create_access_token


class AuthService:

    def __init__(self, db: Session):
        self.user_repository = UserRepository(db)

    def register(
        self,
        username: str,
        email: str,
        password: str,
    ) -> User:
        """
        Register a new user.
        """

        existing = self.user_repository.get_by_email(email)

        if existing:
            raise ValueError(
                "Email already registered."
            )

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
        )

        return self.user_repository.create(user)

    def login(
        self,
        email: str,
        password: str,
    ) -> str:
        """
        Authenticate a user and return a JWT token.
        """

        user = self.user_repository.get_by_email(email)

        if user is not None and not user.is_active:
            raise ValueError(
                "This account has been deactivated."
            )

        if (
            user is None
            or not verify_password(
                password,
                user.password_hash,
            )
        ):
            raise ValueError(
                "Invalid email or password."
            )

        return create_access_token(
            {
                "sub": str(user.id),
            }
        )

    def get_current_user(
        self,
        user_id: UUID,
    ) -> User | None:
        """
        Retrieve the authenticated user.
        """

        return self.user_repository.get_by_id(user_id)
    

    def update_profile(
    self,
    user: User,
    username: str,
    email: str,
) -> User:
        """
        Update the authenticated user's profile.
        """

        existing = self.user_repository.get_by_email(email)

        if existing and existing.id != user.id:
            raise ValueError(
                "Email already registered."
            )

        user.username = username
        user.email = email

        return self.user_repository.update(user)
    

    def change_password(
    self,
    user: User,
    current_password: str,
    new_password: str,
) -> User:
        """
        Change the authenticated user's password.
        """

        if not verify_password(
            current_password,
            user.password_hash,
        ):
            raise ValueError(
                "Current password is incorrect."
            )

        if verify_password(
            new_password,
            user.password_hash,
        ):
            raise ValueError(
                "New password must be different from the current password."
            )
        

        user.password_hash = hash_password(
            new_password,
        )

        return self.user_repository.update(user)
    

    def deactivate_account(
    self,
    user: User,
) -> User:
        """
        Soft delete the authenticated user's account.
        """

        return self.user_repository.deactivate(user)