__version__ = "0.7.2"

import logging

from .base import LOGGER


def set_logger(logger: logging.Logger):
    from . import base
    from . import factor
    from . import decision_core
    from . import strategy

    base.set_logger(logger=logger)
    factor.set_logger(logger=LOGGER.getChild('Factor'))
    decision_core.set_logger(logger=LOGGER.getChild('DecisionCore'))
    strategy.set_logger(logger=LOGGER.getChild('Strategy'))


LOGGER.info(f'Quark version {__version__}!')