__version__ = "0.2.2"

import logging

from .base import LOGGER


def set_logger(logger: logging.Logger):
    base.set_logger(logger=logger)
    factor.set_logger(logger=LOGGER.getChild('Factor'))
    decision_core.set_logger(logger=LOGGER.getChild('DecisionCore'))
    strategy.set_logger(logger=LOGGER.getChild('Strategy'))


from . import base
from . import factor
from . import decision_core
from . import strategy

try:
    from . import calibration
    from . import datalore
except ImportError as _:
    LOGGER.error(f'{_}')
    LOGGER.warning('Quark Datalore model not found! Please install Quark-Calibration to use calibration function!')
