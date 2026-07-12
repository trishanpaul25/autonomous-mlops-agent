from server.exceptions.base import AutonomousMLOpsException


class DatasetNotFoundException(AutonomousMLOpsException):
    def __init__(self):
        super().__init__(
            message="Dataset not found.",
            status_code=404,
            error_code="DATASET_NOT_FOUND"
        )


class UnsupportedFileTypeException(AutonomousMLOpsException):
    def __init__(self):
        super().__init__(
            message="Unsupported file type.",
            status_code=400,
            error_code="UNSUPPORTED_FILE_TYPE"
        )


class EmptyDatasetException(AutonomousMLOpsException):
    def __init__(self):
        super().__init__(
            message="Uploaded dataset is empty.",
            status_code=400,
            error_code="EMPTY_DATASET"
        )


class DatasetReadException(AutonomousMLOpsException):
    def __init__(self):
        super().__init__(
            message="Unable to read dataset.",
            status_code=400,
            error_code="DATASET_READ_ERROR"
        )