import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('API')


def set_logger(logger: logging.Logger):
    import historical
    import historical_xtp

    historical.LOGGER = logger
    historical_xtp.LOGGER = logger


from . import external
from . import historical
from . import historical_xtp
from . import utils
