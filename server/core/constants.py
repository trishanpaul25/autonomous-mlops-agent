from enum import Enum


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    WAITING_FOR_USER = "WAITING_FOR_USER"


class FileType(str, Enum):
    CSV = ".csv"
    EXCEL = ".xlsx"
    JSON = ".json"
    ZIP = ".zip"

class ErrorCode(str, Enum):
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    UNSUPPORTED_FILE_TYPE = "UNSUPPORTED_FILE_TYPE"
    EMPTY_DATASET = "EMPTY_DATASET"


DEFAULT_ENCODING = "utf-8"

DEFAULT_RANDOM_STATE = 42

MAX_CHAT_HISTORY = 100

LOGGER_NAME = "autonomous_mlops"