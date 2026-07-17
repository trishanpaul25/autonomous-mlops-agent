from sqlalchemy.orm import Session

from server.models.trained_model import TrainedModel


class ModelRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        trained_model: TrainedModel,
    ) -> TrainedModel:

        self.db.add(trained_model)

        self.db.commit()

        self.db.refresh(trained_model)

        return trained_model