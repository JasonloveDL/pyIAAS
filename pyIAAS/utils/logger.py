import logging
from logging.handlers import RotatingFileHandler

default_level = logging.INFO  # set log level
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(default_level)
console.setFormatter(formatter)


def get_logger(name, log_file):
    """
    get a logger
    :param name: name of running instance
    """
    handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 2, backupCount=10)
    handler.setLevel(default_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger_ = logging.getLogger(name)
    logger_.setLevel(level=default_level)
    logger_.addHandler(handler)
    logger_.addHandler(console)
    return logger_


if __name__ == '__main__':
    # test suites
    logger = get_logger('server')
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
