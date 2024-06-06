import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('API')


def set_logger(logger: logging.Logger):
    import historical
    import historical_xtp

    historical.LOGGER = logger
    historical_xtp.LOGGER = logger


import external
import historical
import historical_xtp
import utils
