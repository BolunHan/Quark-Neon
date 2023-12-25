import abc

import numpy as np

from .. import LOGGER

__all__ = ['LOGGER']

LOGGER = LOGGER.getChild('Calibration')


class Regression(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x: list | np.ndarray, y: list | np.ndarray, **kwargs) -> None: ...

    @abc.abstractmethod
    def predict(self, x: list | np.ndarray, alpha=0.05) -> tuple[float, tuple[float, float], ...]: ...
    """
    returns a prediction value, 
    and a prediction interval (lower bound and upper bound, of confidence interval {alpha}, minus the prediction value) 
    and other relevant values.
    """

    @abc.abstractmethod
    def to_json(self, fmt='dict') -> dict | str: ...

    @classmethod
    @abc.abstractmethod
    def from_json(cls, json_str: str | bytes | dict): ...
