from enum import Enum


class PipelineStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class FileType(str, Enum):
    CSV = ".csv"
    EXCEL = ".xlsx"
    JSON = ".json"
    ZIP = ".zip"


DEFAULT_ENCODING = "utf-8"

DEFAULT_RANDOM_STATE = 42

MAX_CHAT_HISTORY = 100

LOGGER_NAME = "autonomous_mlops"