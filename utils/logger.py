"""
Centralized logger for the Autonomous MLOps Framework.
"""

import logging
from pathlib import Path


# Resolve logs/ relative to this file (utils/logger.py → project_root/logs/)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "autonomous_mlops.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)


logger = logging.getLogger("AutonomousMLOps")