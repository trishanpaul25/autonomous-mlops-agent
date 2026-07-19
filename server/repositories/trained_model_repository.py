from uuid import UUID

from sqlalchemy.orm import Session

from server.models.trained_model import TrainedModel


class TrainedModelRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, trained_model: TrainedModel) -> TrainedModel:
        """
        Save a trained model.
        """
        self.db.add(trained_model)
        self.db.commit()
        self.db.refresh(trained_model)

        return trained_model

    def create_many(
        self,
        trained_models: list[TrainedModel],
    ) -> None:
        """
        Save multiple trained models.
        """
        self.db.add_all(trained_models)
        self.db.commit()

    def get_by_id(self, model_id: UUID) -> TrainedModel | None:
        """
        Fetch a trained model.
        """
        return (
            self.db.query(TrainedModel)
            .filter(TrainedModel.id == model_id)
            .first()
        )
    
    def get_by_run_id(self, run_id: UUID) -> list[TrainedModel]:
        """
        Fetch all trained models for a pipeline run.
        """
        return (
            self.db.query(TrainedModel)
            .filter(TrainedModel.run_id == run_id)
            .all()
        )

    def update(self, trained_model: TrainedModel) -> TrainedModel:
        self.db.commit()
        self.db.refresh(trained_model)
        return trained_model