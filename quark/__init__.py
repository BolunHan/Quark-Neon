__version__ = "0.2.0"

import logging

from .base import LOGGER


def set_logger(logger: logging.Logger):
    base.set_logger(logger=logger)
    api.set_logger(logger=LOGGER.getChild('API'))
    factor.set_logger(logger=LOGGER.getChild('Factor'))
    decision_core.set_logger(logger=LOGGER.getChild('DecisionCore'))
    strategy.set_logger(logger=LOGGER.getChild('Strategy'))

import base
import api
import factor
import decision_core
import strategy
