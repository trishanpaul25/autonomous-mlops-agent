from uuid import UUID

from sqlalchemy.orm import Session

from server.models.user import User


class UserRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, user: User) -> User:
        """
        Save a new user.
        """
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user

    def get_by_id(
        self,
        user_id: UUID,
    ) -> User | None:
        """
        Fetch a user by ID.
        """
        return (
            self.db.query(User)
            .filter(User.id == user_id)
            .first()
        )

    def get_by_email(
        self,
        email: str,
    ) -> User | None:
        """
        Fetch a user by email.
        """
        return (
            self.db.query(User)
            .filter(User.email == email)
            .first()
        )

    def update(
        self,
        user: User,
    ) -> User:
        """
        Update an existing user.
        """
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        return user