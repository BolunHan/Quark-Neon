import logging

from ..base import LOGGER

LOGGER = LOGGER.getChild('Apps')


def set_logger(logger: logging.Logger):
    global LOGGER
    LOGGER = logger

    from . import web_ui
    web_ui.set_logger(LOGGER.getChild('Backtester'))
