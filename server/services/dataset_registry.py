"""
Temporary dataset registry.

Later this will be replaced by PostgreSQL.
"""


class DatasetRegistry:
    """
    Stores uploaded dataset metadata in memory.

    Later this registry will be replaced by PostgreSQL.
    """

    def __init__(self):
        self.datasets: dict[str, dict] = {}

    def add_dataset(
        self,
        dataset_info: dict,
    ) -> None:
        """
        Register an uploaded dataset.
        """
        self.datasets[dataset_info["dataset_id"]] = dataset_info

    def get_dataset(
        self,
        dataset_id: str,
    ) -> dict | None:
        """
        Retrieve uploaded dataset metadata.
        """
        return self.datasets.get(dataset_id)


dataset_registry = DatasetRegistry()