import json
from typing import Self

import numpy as np

from .utils import EMA

__all__ = ['MACD']


class MACD(object):
    """
    This model calculates the MACD absolute value (not the relative / adjusted value)

    use update_macd method to update the close price
    assign memory array with shm.buf to use shared memory features.
    """

    def __init__(self, short_window=12, long_window=26, signal_window=9, memory_array: list[float] | np.ndarray = None):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

        self.memory_array = memory_array if memory_array is not None else np.array([np.nan] * 6)

    @classmethod
    def update_ema(cls, value: float, memory: float, window: int = None, alpha: float = None):
        return EMA.calculate_ema(value=value, memory=memory, window=window, alpha=alpha)

    def calculate_macd(self, price: float) -> dict[str, float]:
        self.price = price
        ema_short = self.ema_short if np.isfinite(self.ema_short) else price
        ema_long = self.ema_long if np.isfinite(self.ema_long) else price

        ema_short = self.update_ema(value=price, memory=ema_short, window=self.short_window)
        ema_long = self.update_ema(value=price, memory=ema_long, window=self.long_window)

        macd_line = ema_short - ema_long

        signal_line = self.signal_line if np.isfinite(self.signal_line) else macd_line

        signal_line = self.update_ema(value=macd_line, memory=signal_line, window=self.signal_window)
        macd_diff = macd_line - signal_line

        return dict(
            ema_short=ema_short,
            ema_long=ema_long,
            macd_line=macd_line,
            signal_line=signal_line,
            macd_diff=macd_diff
        )

    def update_macd(self, price: float) -> dict[str, float]:
        macd_dict = self.calculate_macd(price=price)

        self.ema_short = macd_dict['ema_short']
        self.ema_long = macd_dict['ema_long']
        self.macd_line = macd_dict['macd_line']
        self.signal_line = macd_dict['signal_line']
        self.macd_diff = macd_dict['macd_diff']

        return macd_dict

    def get_macd_values(self) -> dict[str, float]:
        return dict(
            ema_short=self.ema_short,
            ema_long=self.ema_long,
            macd_line=self.macd_line,
            signal_line=self.signal_line,
            macd_diff=self.macd_diff
        )

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            short_window=self.short_window,
            long_window=self.long_window,
            signal_window=self.signal_window,
            memory_array=self.memory_array.tolist()
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            short_window=json_dict['short_window'],
            long_window=json_dict['long_window'],
            signal_window=json_dict['signal_window'],
            memory_array=np.array(json_dict['memory_array'])
        )

        return self

    def clear(self):
        self.ema_short = np.nan
        self.ema_long = np.nan

        self.macd_line = np.nan
        self.signal_line = np.nan
        self.macd_diff = np.nan
        self.price = np.nan

    @property
    def ema_short(self) -> float:
        return self.memory_array[0]

    @ema_short.setter
    def ema_short(self, value: float):
        self.memory_array[0] = value

    @property
    def ema_long(self) -> float:
        return self.memory_array[1]

    @ema_long.setter
    def ema_long(self, value: float):
        self.memory_array[1] = value

    @property
    def macd_line(self) -> float:
        return self.memory_array[2]

    @macd_line.setter
    def macd_line(self, value: float):
        self.memory_array[2] = value

    @property
    def signal_line(self) -> float:
        return self.memory_array[3]

    @signal_line.setter
    def signal_line(self, value: float):
        self.memory_array[3] = value

    @property
    def macd_diff(self) -> float:
        return self.memory_array[4]

    @macd_diff.setter
    def macd_diff(self, value: float):
        self.memory_array[4] = value

    @property
    def price(self) -> float:
        return self.memory_array[5]

    @price.setter
    def price(self, value: float):
        self.memory_array[5] = value

    @property
    def macd_diff_adjusted(self):
        return self.macd_diff / self.price
