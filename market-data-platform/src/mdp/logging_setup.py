import logging
import os


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("MDP_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
