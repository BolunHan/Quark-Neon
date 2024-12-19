import logging

from .. import LOGGER


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger
    from . import candlestick
    from . import dashboard

    candlestick.LOGGER = LOGGER
    dashboard.LOGGER = LOGGER


from .candlestick import Candlestick
from .dashboard import Dashboard
from .state_tracker import StateBanner

__all__ = ['LOGGER', 'Candlestick', 'Dashboard', 'StateBanner']
