import abc

import numpy as np
import pandas as pd

from .. import LOGGER

LOGGER = LOGGER.getChild('DataLore')


class DataLore(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def calibrate(self, factor_value: pd.DataFrame | list[pd.DataFrame], **kwargs):
        ...

    @abc.abstractmethod
    def predict(self, factor_value: dict[str, float], **kwargs) -> dict[str, float]:
        ...

    @abc.abstractmethod
    def predict_batch(self, factor_value: pd.DataFrame | dict[str, list[float] | np.ndarray], **kwargs) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def clear(self):
        ...

    @abc.abstractmethod
    def to_json(self, fmt='dict') -> dict | str:
        ...

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        type_str = json_str['type']

        if type_str == 'LinearDataLore':
            from .data_lore import LinearDataLore
            return LinearDataLore.from_json(json_str)
        else:
            raise NotImplementedError(f'Invalid data lore type {type_str}.')

    @property
    def is_ready(self):
        return True
