from enum import Enum


class ProgressEventType(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    THINKING = "thinking"
    STEP = "step"
    COMPLETE = "complete"