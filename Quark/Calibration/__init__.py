import abc
import json
import pathlib
from typing import overload

import numpy as np

from .. import LOGGER

LOGGER = LOGGER.getChild('Calibration')


class Regression(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x: list | np.ndarray, y: list | np.ndarray, **kwargs) -> None: ...

    @overload
    def predict(self, x: np.ndarray, alpha=0.05) -> tuple[np.ndarray, np.ndarray, ...]: ...

    @abc.abstractmethod
    def predict(self, x: list | np.ndarray, alpha=0.05) -> tuple[float, tuple[float, float], ...]:
        """
        note: alpha is the prob outside the center distribution, e.g. alpha = 0.1 -> lower_bound=0.05, upper_bound=0.95; alpha = 1 -> lower_bound=upper_bound=0.5
        returns a prediction value,
        and a prediction interval (lower bound and upper bound, of confidence interval {alpha}, minus the prediction value)
        and prediction deviation (bootstrapped value - mean) for bootstrapping model and bagging models
        and other relevant values.
        """
        ...

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

    @property
    def ensemble(self) -> str:
        return 'simple'


from .cross_validation import CrossValidation, Metrics

__all__ = ['LOGGER', 'Regression', 'CrossValidation', 'Metrics']
