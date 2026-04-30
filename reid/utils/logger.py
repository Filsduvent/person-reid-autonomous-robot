import logging
import os
import sys


def setup_logger(name: str, output_dir: str, filename: str = "log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    log_path = os.path.abspath(os.path.join(output_dir, filename))

    has_stdout_handler = False
    has_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            has_stdout_handler = True
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == log_path:
            has_file_handler = True

    if not has_stdout_handler:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not has_file_handler:
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
