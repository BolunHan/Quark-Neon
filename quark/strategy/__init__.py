import logging

from algo_engine.engine import MDS
from algo_engine.strategy import STRATEGY_ENGINE

from .. import LOGGER

LOGGER = LOGGER.getChild('Strategy')


def set_logger(logger: logging.Logger):
    from . import strategy

    strategy.LOGGER = logger.getChild('Strategy')


from .strategy import *

__all__ = [
    # Basic
    'LOGGER', 'STRATEGY_ENGINE', 'MDS',
    # names from .strategy
    'StatusCode', 'Strategy',
]
