from logging import Logger
import logging
import os


def create_logger(working_path: str, level: str = logging.INFO) -> Logger:
    logger = logging.getLogger()
    logger.setLevel(level)

    log_file = os.path.join(f"{working_path}/run.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
