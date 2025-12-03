import logging
import sys
from logging import StreamHandler, Formatter

def get_logger(name: str = __name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

# convenience top-level logger
logging = get_logger("rag_project")
