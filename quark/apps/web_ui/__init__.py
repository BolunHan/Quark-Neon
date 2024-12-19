import logging

from .. import LOGGER

LOGGER = LOGGER.getChild('Backtester')

from .web_app import WebApp, start_app
from .tester import Tester


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger
    from . import docserver

    web_app.LOGGER = LOGGER
    docserver.set_logger(LOGGER)


__all__ = ['WebApp', 'start_app', 'Tester']
