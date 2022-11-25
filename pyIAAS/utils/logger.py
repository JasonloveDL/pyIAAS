import logging

from concurrent_log_handler import ConcurrentRotatingFileHandler


def get_logger(name, log_file, level=logging.INFO):
    """
    get a logger
    :param name: name of running instance
    """
    logger_ = logging.getLogger(name)
    if len(logger_.handlers) != 0:
        logger_.handlers.clear()

    handler = ConcurrentRotatingFileHandler(log_file, maxBytes=1024 * 1024 * 2, backupCount=10)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger_.addHandler(handler)
    logger_.addHandler(console)
    logger_.setLevel(level=level)
    return logger_
