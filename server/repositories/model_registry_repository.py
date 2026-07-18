from uuid import UUID

from sqlalchemy.orm import Session

from server.models.model_registry import ModelRegistry


class ModelRegistryRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        model: ModelRegistry,
    ) -> ModelRegistry:
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)

        return model

    def get_by_run_id(
        self,
        run_id: UUID,
    ) -> list[ModelRegistry]:
        return (
            self.db.query(ModelRegistry)
            .filter(ModelRegistry.run_id == run_id)
            .all()
        )