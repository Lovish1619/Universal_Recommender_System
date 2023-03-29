import logging
from datetime import datetime
import os

# Declaring constants regarding log files
LOG_DIR = "recommender_logs"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Making log directory
os.makedirs(LOG_DIR, exist_ok=True)

# Setting configurations for logging file
logging.basicConfig(
    filename = LOG_FILE_PATH,
    filemode = "w",
    format = "[%(sctime)s] %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)