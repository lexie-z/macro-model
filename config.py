import logging
import os

log_file = "kalman_checks.log"

if os.path.exists(log_file):
    with open(log_file, "w"):
        pass  

# Set up logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='a'),  
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
