import logging

from quark import LOGGER

LOGGER = LOGGER.getChild('DecisionCore')

from .decision_core import StrategyProfile, StrategyState, DecisionCore, DummyDecisionCore


def set_logger(logger: logging.Logger):
    from . import decision_core

    decision_core.LOGGER = logger


__all__ = ['StrategyProfile', 'StrategyState', 'DecisionCore', 'DummyDecisionCore']
