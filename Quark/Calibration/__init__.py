import abc
import json
import pathlib

import numpy as np
from typing import overload
from .. import LOGGER

__all__ = ['LOGGER']

LOGGER = LOGGER.getChild('Calibration')


class Regression(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x: list | np.ndarray, y: list | np.ndarray, **kwargs) -> None: ...

    @overload
    def predict(self, x: np.ndarray, alpha=0.05) -> tuple[np.ndarray, np.ndarray, ...]: ...

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

    def dump(self, file_path: str | pathlib.Path = None):
        json_dict = self.to_json(fmt='dict')
        json_str = json.dumps(json_dict, indent=4)

        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(json_str)

        return json_dict
