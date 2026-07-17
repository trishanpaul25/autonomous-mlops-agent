from uuid import UUID

from sqlalchemy.orm import Session

from server.models.dataset import Dataset


class DatasetRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, dataset: Dataset) -> Dataset:
        """
        Save a dataset in PostgreSQL.
        """
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)

        return dataset

    def get_by_id(self, dataset_id: UUID) -> Dataset | None:
        """
        Fetch a dataset by ID.
        """
        return (
            self.db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

    def get_all(self) -> list[Dataset]:
        """
        Fetch all active datasets.
        """
        return (
            self.db.query(Dataset)
            .filter(Dataset.is_deleted == False)
            .all()
        )
    
    def get_by_user(self, user_id: UUID) -> list[Dataset]:
        """
        Fetch all datasets uploaded by a user.
        """
        return (
            self.db.query(Dataset)
            .filter(
                Dataset.user_id == user_id,
                Dataset.is_deleted == False,
            )
            .all()
        )

    def get_by_path(self, dataset_path: str) -> Dataset | None:
        """
        Fetch a dataset by path.
        """
        return (
            self.db.query(Dataset)
            .filter(Dataset.dataset_path == dataset_path)
            .first()
        )

    def delete(self, dataset: Dataset) -> None:
        """
        soft Delete a dataset.
        """
        dataset.is_deleted = True
        self.db.commit()


    def restore(self, dataset: Dataset) -> None:
        """
        Restore a soft deleted dataset.
        """
        dataset.is_deleted = False
        self.db.commit()