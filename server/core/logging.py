import logging
from logging.handlers import RotatingFileHandler
from server.core.constants import LOGGER_NAME
from server.core.config import settings

from server.core.paths import LOG_DIR

LOG_FILE = LOG_DIR/"server.log"

def setup_logger() -> logging.Logger:
    """
    Configure and return the application logger.
    """

    #retrieves or creates a logger instance 
    logger = logging.getLogger(LOGGER_NAME)

    #duplicate handler check  
    if logger.hasHandlers():
        return logger

    #DEBUG ignored 
    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    #creates handler to print log messages directly in the terminal console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    #activating the loggers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger

def get_logger(name: str) -> logging.Logger:
    #logger for specific modules
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

def initialize_logging():
    setup_logger()