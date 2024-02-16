from __future__ import annotations

import abc
import enum
import json
import pickle
from collections import deque

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData, TickData, BarData

from .. import LOGGER
from ..Calibration.dummies import is_market_session
from multiprocessing import shared_memory, Lock

__all__ = ['IndexWeight', 'EMA', 'MACD', 'Synthetic', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler']


class SharedMemoryCore(object):
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def init_buffer(self, name: str, buffer_size: int) -> shared_memory.SharedMemory:
        try:
            shm = shared_memory.SharedMemory(name=name)

            if shm.size != buffer_size:
                shm.close()
                shm.unlink()

                shm = shared_memory.SharedMemory(name=name, create=True, size=buffer_size)
        except FileNotFoundError as _:
            shm = shared_memory.SharedMemory(name=name, create=True, size=buffer_size)

        return shm

    def get_buffer(self, name: str) -> shared_memory.SharedMemory | None:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError as _:
            shm = None

        return shm

    def set_int(self, name: str, value: int, buffer_size: int = 32) -> shared_memory.SharedMemory:
        shm = self.init_buffer(name=name, buffer_size=buffer_size)
        shm.buf[:] = value.to_bytes(length=buffer_size)
        shm.close()
        return shm

    def get_int(self, name: str, default: int = None) -> int | None:
        shm = self.get_buffer(name=name)
        value = default if shm is None else int.from_bytes(shm.buf)
        shm.close()
        return value

    def set_vector(self, name: str, vector: list[float | int | bool] | np.ndarray) -> shared_memory.SharedMemory:
        arr = np.array(vector)
        shm = self.init_buffer(name=name, buffer_size=arr.nbytes)
        shm.buf[:] = bytes(arr.data)
        shm.close()
        return shm

    def get_vector(self, name: str, target: list = None) -> list[float]:
        shm = self.get_buffer(name=name)
        vector = [] if shm is None else np.ndarray(shape=(-1,), buffer=shm.buf).tolist()
        shm.close()

        if target is None:
            return vector
        else:
            target.clear()
            target.extend(vector)
            return target

    def set_str_vector(self, name: str, vector: list[str]) -> None:
        vector_bytes = b'\x00'.join([_.encode(self.encoding) for _ in vector])
        shm = self.init_buffer(name=name, buffer_size=len(vector_bytes))
        shm.buf[:] = vector_bytes
        shm.close()

    def get_str_vector(self, name: str, target: list = None) -> list[float]:
        shm = self.get_buffer(name=name)
        vector = [] if shm is None else [_.decode(self.encoding) for _ in bytes(shm.buf).split(b'\x00')]
        shm.close()

        if target is None:
            return vector
        else:
            target.clear()
            target.extend(vector)
            return target

    def set_named_vector(self, name: str, obj: dict[str, float]) -> None:
        """
        sync the dict[str, float]
        """
        keys_name = f'{name}.keys'
        values_name = f'{name}.values'

        self.set_str_vector(name=keys_name, vector=list(obj.keys()))
        self.set_vector(name=values_name, vector=list(obj.values()))

    def get_name_vector(self, name: str, target: dict = None) -> dict[str, float]:
        keys_name = f'{name}.keys'
        values_name = f'{name}.values'

        keys = self.get_str_vector(name=keys_name)
        values = self.get_vector(name=values_name)
        shm_dict = dict(zip(keys, values))

        if target is None:
            return shm_dict
        else:
            target.clear()
            target.update(shm_dict)
            return target

    def sync(self, name: str, obj: ...) -> shared_memory.SharedMemory:
        serialized = pickle.dumps(obj)
        shm = self.init_buffer(name=name, buffer_size=len(serialized))
        shm.buf[:] = serialized
        shm.close()

        return shm

    def get(self, name: str, default=None) -> ...:
        shm = self.get_buffer(name=name)
        obj = default if shm is None else pickle.loads(bytes(shm.buf))
        shm.close()

        return obj


class CachedMemoryCore(SharedMemoryCore):
    def __init__(self, prefix: str, encoding='utf-8'):
        super().__init__(encoding=encoding)

        self.prefix = prefix

        self.shm_size: dict[str, shared_memory.SharedMemory] = {}
        self.shm_cache: dict[str, shared_memory.SharedMemory] = {}

    def get_buffer(self, name: str = None, real_name: str = None) -> shared_memory.SharedMemory | None:

        if real_name is None and name is None:
            raise ValueError('Must assign a "name" or "real_name",')
        elif real_name is None:
            real_name = f'{self.prefix}.{name}'

        # get the shm storing the size data
        if real_name in self.shm_size:
            shm_size = self.shm_size[real_name]
        else:
            shm_size = super().get_buffer(name=f'{real_name}.size')

            # no size info found
            # since the get_buffer should be called after init_buffer, thus the shm_size should be initialized before this function called.
            # this should not happen. an error message is generated.
            # no cache info stored
            if shm_size is None:
                LOGGER.error(f'Shared memory "{real_name}.size" not found! Expect a 8 bytes shared memory.')
                return super().get_buffer(name=real_name)

            self.shm_size[real_name] = shm_size

        cache_size = int.from_bytes(shm_size.buf)
        shm_size.close()

        # cache not hit. This could happen in the child-processes.
        if name not in self.shm_cache:
            shm = super().get_buffer(name=real_name)

            # for the similar reason above, the get_buffer should be called after init_buffer
            # shm should never be None
            if shm is None:
                LOGGER.error(f'Shared memory "{real_name}" not found!, you should call init_buffer first')
            else:
                self.shm_cache[real_name] = shm

            return shm
        shm = self.shm_cache[real_name]

        # the cache size is the requested size, cache validated.
        if shm.size == cache_size:
            return shm

        # the cache size does not match the requested size, cache validation failed, this could be a result of lack of lock.
        shm = super().get_buffer(name=real_name)
        # the get-process should not update the size log, this is the tasks for the process altering shared memory.
        # shm_size.buf[:] = shm.size.to_bytes(length=8)
        return shm

    def init_buffer(self, buffer_size: int, name: str = None, real_name: str = None) -> shared_memory.SharedMemory:
        if real_name is None and name is None:
            raise ValueError('Must assign a "name" or "real_name",')
        elif real_name is None:
            real_name = f'{self.prefix}.{name}'

        # cache size log found in local
        if real_name in self.shm_size:
            shm_size = self.shm_size[real_name]
            cache_size = int.from_bytes(shm_size.buf)

            # since the cache size info exist, the shm must exist, in ether memory or local or both.
            if real_name in self.shm_cache:
                shm = self.shm_cache[real_name]
            else:
                shm = self.shm_cache[real_name] = shared_memory.SharedMemory(name=real_name)

            # cache hit
            if shm.size == buffer_size == cache_size:
                shm_size.close()
                return shm

            # cache not hit, unlink the original shm and create a new one
            shm.close()
            shm.unlink()
            shm_size.buf[:] = buffer_size.to_bytes(length=8)
            shm_size.close()
            shm = self.shm_cache[real_name] = shared_memory.SharedMemory(name=real_name, create=True, size=buffer_size)
            return shm
        # cache size info not found
        elif (shm_size := super().get_buffer(name=f'{real_name}.size')) is None:
            self.shm_size[real_name] = shm_size = shared_memory.SharedMemory(name=f'{real_name}.size', create=True, size=8)
            shm_size.buf[:] = buffer_size.to_bytes(length=8)
            shm_size.close()

            # no cache size info but still have a cached obj, this should never happen
            if real_name in self.shm_cache:
                raise ValueError('Cache found but no cache size info found, potential collision on shared memory names, stop and exit is advised.')

            shm = self.shm_cache[real_name] = shared_memory.SharedMemory(name=real_name, create=True, size=buffer_size)
            return shm
        # cache size found in memory: update local logs and re-run
        else:
            self.shm_size[real_name] = shm_size
            # avoid issues in nested-inheritance
            return CachedMemoryCore.init_buffer(self=self, real_name=real_name, buffer_size=buffer_size)


class SyncMemoryCore(CachedMemoryCore):
    def __init__(self, prefix: str, encoding='utf-8'):
        super().__init__(prefix=prefix, encoding=encoding)
        self.lock = Lock()

    def get_buffer(self, name: str = None, real_name: str = None) -> shared_memory.SharedMemory | None:
        with self.lock:
            return super().get_buffer(name=name, real_name=real_name)

    def init_buffer(self, buffer_size: int, name: str = None, real_name: str = None, ) -> shared_memory.SharedMemory:
        with self.lock:
            return super().init_buffer(name=name, real_name=real_name, buffer_size=buffer_size)



class IndexWeight(dict):
    def __init__(self, index_name: str, *args, **kwargs):
        self.index_name = index_name

        super().__init__(*args, **kwargs)

    def normalize(self):
        total_weight = sum(list(self.values()))

        if not total_weight:
            return

        for _ in self:
            self[_] /= total_weight

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        weighted_sum = 0.

        for ticker, weight in self.items():
            value = values.get(ticker, replace_na)

            if np.isnan(value):
                weighted_sum += replace_na * self[ticker]
            else:
                weighted_sum += value * self[ticker]

        return weighted_sum

    @property
    def components(self) -> list[str]:
        return list(self.keys())

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.index_name,
            weights=dict(self),

        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> IndexWeight:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            index_name=json_dict['index_name'],
            **json_dict['weights']
        )

        return self


class EMA(object):
    def __init__(self, discount_interval: float, alpha: float = None, window: int = None):
        self.discount_interval = discount_interval
        self.alpha = alpha if alpha else 1 - 2 / (window + 1)
        self.window = window if window else round(2 / (1 - alpha) - 1)

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.__class__.__name__} should have an alpha from 0 to 1')

        if discount_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive discount_interval')

        self._last_discount_ts: dict[str, float] = {}
        self._history: dict[str, dict[str, float]] = getattr(self, '_history', {})
        self._current: dict[str, dict[str, float]] = getattr(self, '_current', {})
        self._window: dict[str, dict[str, deque[float]]] = getattr(self, '_window', {})
        self.ema: dict[str, dict[str, float]] = getattr(self, 'ema', {})

    @classmethod
    def calculate_ema(cls, value: float, memory: float = None, window: int = None, alpha: float = None):
        if alpha is None and window is None:
            raise ValueError('Must assign value to alpha or window.')
        elif alpha is None:
            alpha = 2 / (window + 1)

        if memory is None:
            return value

        ema = alpha * value + (1 - alpha) * memory
        return ema

    def register_ema(self, name):
        self._history[name] = {}
        self._current[name] = {}
        self._window[name] = {}
        _ = self.ema[name] = {}
        return _

    def update_ema(self, ticker: str, timestamp: float = None, replace_na: float = np.nan, **update_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {update_data}')
                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = (timestamp // self.discount_interval) * self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)

        # update to current
        for entry_name in update_data:
            if entry_name in self._current:
                if np.isfinite(value := update_data[entry_name]):
                    current = self._current[entry_name][ticker] = value
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)
                    else:
                        self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)

    def accumulate_ema(self, ticker: str, timestamp: float = None, replace_na: float = np.nan, **accumulative_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {accumulative_data}')

                time_span = max(0., last_discount - timestamp)
                adjust_factor = time_span // self.discount_interval
                alpha = self.alpha ** adjust_factor

                for entry_name in accumulative_data:
                    if entry_name in self._history:
                        if np.isfinite(_ := accumulative_data[entry_name]):
                            current = self._current[entry_name].get(ticker, 0.)
                            memory = self._history[entry_name][ticker] = self._history[entry_name].get(ticker, 0.) + _ * alpha

                            if memory is None:
                                self.ema[entry_name][ticker] = replace_na * self.alpha + current * (1 - self.alpha)
                            else:
                                self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = (timestamp // self.discount_interval) * self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)
        # add to current
        for entry_name in accumulative_data:
            if entry_name in self._current:
                if np.isfinite(_ := accumulative_data[entry_name]):
                    current = self._current[entry_name][ticker] = self._current[entry_name].get(ticker, 0.) + _
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)
                    else:
                        self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)

    def discount_ema(self, ticker: str, timestamp: float):
        last_update = self._last_discount_ts.get(ticker, timestamp)

        # a discount event is triggered
        if last_update + self.discount_interval <= timestamp:
            time_span = timestamp - last_update
            adjust_power = int(time_span // self.discount_interval)

            for entry_name in self._history:
                current = self._current[entry_name].get(ticker)
                memory = self._history[entry_name].get(ticker)
                window: deque = self._window[entry_name].get(ticker, deque(maxlen=self.window))

                # pre-check: drop None or nan
                if current is None or not np.isfinite(current):
                    return

                # step 1: update window
                for _ in range(adjust_power - 1):
                    if window:
                        window.append(window[-1])

                window.append(current)

                # step 2: re-calculate memory if window is not full
                if len(window) < window.maxlen or memory is None:
                    memory = None

                    for _ in window:
                        if memory is None:
                            memory = _

                        memory = memory * self.alpha + _ * (1 - self.alpha)

                # step 3: calculate ema value by memory and current value
                ema = memory * self.alpha + current * (1 - self.alpha)

                # step 4: update EMA state
                self._current[entry_name].pop(ticker)
                self._history[entry_name][ticker] = ema
                self._window[entry_name][ticker] = window
                self.ema[entry_name][ticker] = ema

        self._last_discount_ts[ticker] = (timestamp // self.discount_interval) * self.discount_interval

    def _check_discontinuity(self, timestamp: float, tolerance: int = 1):
        discontinued = []

        for ticker in self._last_discount_ts:
            last_update = self._last_discount_ts[ticker]

            if last_update + (tolerance + 1) * self.discount_interval < timestamp:
                discontinued.append(ticker)

        return discontinued

    def discount_all(self, timestamp: float):
        for _ in self._check_discontinuity(timestamp=timestamp, tolerance=1):
            self.discount_ema(ticker=_, timestamp=timestamp)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            discount_interval=self.discount_interval,
            alpha=self.alpha,
            window=self.window,
            last_discount_ts=self._last_discount_ts,
            history=self._history,
            current=self._current,
            window_deque={entry_name: {ticker: list(dq) for ticker, dq in entry} for entry_name, entry in self._window.items()},
            ema=self.ema,
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> EMA:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            discount_interval=json_dict['discount_interval'],
            alpha=json_dict['alpha'],
            window=json_dict['window']
        )

        self._last_discount_ts = json_dict['last_discount_ts']
        self._history = json_dict['history']
        self._current = json_dict['current']
        self._window = {entry_name: {ticker: deque(data, maxlen=self.window) for ticker, data in entry} for entry_name, entry in json_dict['window_deque'].items()}
        self.ema = json_dict['ema']
        return self

    def clear(self):
        self._last_discount_ts.clear()
        self._history.clear()
        self._current.clear()
        self._window.clear()
        self.ema.clear()


class MACD(object):
    """
    This model calculates the MACD absolute value (not the relative / adjusted value)

    use update_macd method to update the close price
    """

    def __init__(self, short_window=12, long_window=26, signal_window=9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window

        self.ema_short = None
        self.ema_long = None

        self.macd_line = None
        self.signal_line = None
        self.macd_diff = None
        self.price = None

    @classmethod
    def update_ema(cls, value: float, memory: float, window: int = None, alpha: float = None):
        return EMA.calculate_ema(value=value, memory=memory, window=window, alpha=alpha)

    def calculate_macd(self, price: float) -> dict[str, float]:
        self.price = price
        ema_short = price if self.ema_short is None else self.ema_short
        ema_long = price if self.ema_long is None else self.ema_long

        ema_short = self.update_ema(value=price, memory=ema_short, window=self.short_window)
        ema_long = self.update_ema(value=price, memory=ema_long, window=self.long_window)

        macd_line = ema_short - ema_long

        signal_line = macd_line if self.signal_line is None else self.signal_line

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

    @property
    def macd_diff_adjusted(self):
        return self.macd_diff / self.price

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            short_window=self.short_window,
            long_window=self.long_window,
            signal_window=self.signal_window,
            ema_short=self.ema_short,
            ema_long=self.ema_long,
            macd_line=self.macd_line,
            signal_line=self.signal_line,
            macd_diff=self.macd_diff,
            price=self.price
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> MACD:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            short_window=json_dict['short_window'],
            long_window=json_dict['long_window'],
            signal_window=json_dict['signal_window']
        )

        self.ema_short = json_dict['ema_short']
        self.ema_long = json_dict['ema_long']
        self.macd_line = json_dict['macd_line']
        self.signal_line = json_dict['signal_line']
        self.macd_diff = json_dict['macd_diff']
        self.price = json_dict['price']

        return self

    def clear(self):
        self.ema_short = None
        self.ema_long = None

        self.macd_line = None
        self.signal_line = None
        self.macd_diff = None
        self.price = None


class Synthetic(object, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float]):
        self.weights: IndexWeight = weights if isinstance(weights, IndexWeight) else IndexWeight(index_name='synthetic', **weights)
        self.weights.normalize()

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price = 1.

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        return self.weights.composite(values=values, replace_na=replace_na)

    def update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        if ticker not in self.base_price:
            self.base_price[ticker] = market_price

        self.last_price[ticker] = market_price

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.weights.index_name,
            weights=dict(self.weights),
            base_price=self.base_price,
            last_price=self.last_price,
            synthetic_base_price=self.synthetic_base_price
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Synthetic:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        weights = IndexWeight(index_name=json_dict['index_name'], **json_dict['weights'])

        self = cls(
            weights=weights
        )

        self.base_price = json_dict['base_price']
        self.last_price = json_dict['last_price']
        self.synthetic_base_price = json_dict['synthetic_base_price']

        return self

    def clear(self):
        self.base_price.clear()
        self.last_price.clear()

    @property
    def synthetic_index(self):
        price_list = []
        weight_list = []

        for ticker in self.weights:
            weight_list.append(self.weights[ticker])

            if ticker in self.last_price:
                price_list.append(self.last_price[ticker] / self.base_price[ticker])
            else:
                price_list.append(1.)

        synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(self.last_price)


class FixedIntervalSampler(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for a fixed interval sampler designed for financial usage.

    Args:
    - sampling_interval (float): Time interval between consecutive samples. Default is 1.
    - sample_size (int): Number of samples to be stored. Default is 60.

    Attributes:
    - sampling_interval (float): Time interval between consecutive samples.
    - sample_size (int): Number of samples to be stored.

    Warnings:
    - If `sampling_interval` is not positive, a warning is issued.
    - If `sample_size` is less than 2, a warning is issued according to Shannon's Theorem.

    Methods:
    - log_obs(ticker: str, value: float, timestamp: float, storage: Dict[str, Dict[float, float]])
        Logs an observation for the given ticker at the specified timestamp.

    - on_entry_added(ticker: str, key, value)
        Callback method triggered when a new entry is added.

    - on_entry_updated(ticker: str, key, value)
        Callback method triggered when an existing entry is updated.

    - on_entry_removed(ticker: str, key, value)
        Callback method triggered when an entry is removed.

    - clear()
        Clears all stored data.

    Notes:
    - Subclasses must implement the abstract methods: on_entry_added, on_entry_updated, on_entry_removed.

    """

    class Mode(enum.Enum):
        update = 'update'
        accumulate = 'accumulate'

    def __init__(self, sampling_interval: float = 1., sample_size: int = 60):
        """
        Initialize the FixedIntervalSampler.

        Parameters:
        - sampling_interval (float): Time interval between consecutive samples (in seconds). Default is 1.
        - sample_size (int): Number of samples to be stored. Default is 60.
        """
        self.sampling_interval = sampling_interval
        self.sample_size = sample_size

        self.sample_storage = getattr(self, 'sample_storage', {})

        # Warning for sampling_interval
        if sampling_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive sampling_interval')

        # Warning for sample_interval by Shannon's Theorem
        if sample_size <= 2:
            LOGGER.warning(f"{self.__class__.__name__} should have a larger sample_size, by Shannon's Theorem, sample_size should be greater than 2")

    def register_sampler(self, name: str, mode: str | Mode = 'update'):
        if name in self.sample_storage:
            raise ValueError(f'name {name} already registered in {self.__class__.__name__}!')

        if isinstance(mode, self.Mode):
            mode = mode.value

        if mode not in ['update', 'accumulate']:
            raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

        self.sample_storage[name] = dict(
            storage={},
            mode=mode
        )

    def get_sampler(self, name: str) -> dict[str, dict]:
        if name not in self.sample_storage:
            raise ValueError(f'name {name} not found in {self.__class__.__name__}!')

        return self.sample_storage[name]['storage']

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        idx = timestamp // self.sampling_interval
        idx_min = idx - self.sample_size

        for obs_name, obs_value in observation_copy.items():
            if obs_name not in self.sample_storage:
                raise ValueError(f'Invalid observation name {obs_name}')

            sampler = self.sample_storage[obs_name]
            storage = sampler['storage']
            mode = sampler['mode']

            if ticker in storage:
                obs_storage = storage[ticker]
            else:
                obs_storage = storage[ticker] = {}

            if idx not in obs_storage:
                obs_storage[idx] = obs_value
                self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
            else:
                if mode == 'update':
                    obs_storage[idx] = obs_value
                elif mode == 'accumulate':
                    obs_storage[idx] += obs_value
                else:
                    raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

                self.on_entry_updated(ticker=ticker, name=obs_name, value=obs_storage[idx])

            for idx_pop in list(obs_storage):
                if idx_pop < idx_min:
                    _ = obs_storage.pop(idx_pop)
                    self.on_entry_removed(ticker=ticker, name=obs_name, value=_)
                else:
                    break

    def on_entry_added(self, ticker: str, name: str, value):
        """
        Callback method triggered when a new entry is added.

        Parameters:
        - ticker (str): Ticker symbol for the added entry.
        - key: Key for the added entry.
        - value: Value of the added entry.
        """
        pass

    def on_entry_updated(self, ticker: str, name: str, value):
        """
        Callback method triggered when an existing entry is updated.

        Parameters:
        - ticker (str): Ticker symbol for the updated entry.
        - key: Key for the updated entry.
        - value: Updated value of the entry.
        """
        pass

    def on_entry_removed(self, ticker: str, name: str, value):
        """
        Callback method triggered when an entry is removed.

        Parameters:
        - ticker (str): Ticker symbol for the removed entry.
        - key: Key for the removed entry.
        - value: Value of the removed entry.
        """
        pass

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            sampling_interval=self.sampling_interval,
            sample_size=self.sample_size,
            sample_storage=self.sample_storage
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> FixedIntervalSampler:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        self.sample_storage = json_dict['sample_storage']

        return self

    def clear(self):
        """
        Clears all stored data.
        """
        for name in self.sample_storage:
            self.sample_storage[name]['storage'].clear()

        # using this code will require the sampler to be registered again.
        self.sample_storage.clear()

    def loc_obs(self, name: str, ticker: str, index: int | slice = None) -> float | list[float]:
        sampler = self.get_sampler(name=name)
        observation = sampler.get(ticker, {})

        if index is None:
            return list(observation.values())
        else:
            return list(observation.values())[index]

    def active_obs(self, name: str) -> dict[str, float]:
        sampler = self.get_sampler(name=name)
        last_obs = {}

        for ticker, observation in sampler.items():
            if observation:
                last_obs[ticker] = list(observation.values())[-1]

        return last_obs


class FixedVolumeIntervalSampler(FixedIntervalSampler, metaclass=abc.ABCMeta):
    """
    Concrete implementation of FixedIntervalSampler with fixed volume interval sampling.

    Args:
    - sampling_interval (float): Volume interval between consecutive samples. Default is 100.
    - sample_size (int): Number of samples to be stored. Default is 20.

    Attributes:
    - sampling_interval (float): Volume interval between consecutive samples.
    - sample_size (int): Number of samples to be stored.
    - _accumulated_volume (Dict[str, float]): Accumulated volume for each ticker.

    Methods:
    - accumulate_volume(ticker: str = None, volume: float = 0., market_data: MarketData = None, use_notional: bool = False)
        Accumulates volume based on market data or explicit ticker and volume.

    - log_obs(ticker: str, value: float, storage: Dict[str, Dict[float, float]], volume_accumulated: float = None)
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

    - clear()
        Clears all stored data, including accumulated volumes.

    Notes:
    - This class extends FixedIntervalSampler and provides additional functionality for volume accumulation.
    - The sampling_interval is in shares, representing the fixed volume interval for sampling.

    """

    def __init__(self, sampling_interval: float = 100., sample_size: int = 20):
        """
        Initialize the FixedVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Volume interval between consecutive samples. Default is 100.
        - sample_size (int): Number of samples to be stored. Default is 20.
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size)
        self._accumulated_volume: dict[str, float] = {}

    def accumulate_volume(self, ticker: str = None, volume: float = 0., market_data: MarketData = None, use_notional: bool = False):
        """
        Accumulates volume based on market data or explicit ticker and volume.

        Parameters:
        - ticker (str): Ticker symbol for volume accumulation.
        - volume (float): Volume to be accumulated.
        - market_data (MarketData): Market data for dynamic volume accumulation.
        - use_notional (bool): Flag indicating whether to use notional instead of volume.

        Raises:
        - NotImplementedError: If market data type is not supported.

        """
        if market_data is not None and (not is_market_session(market_data.timestamp)):
            return

        if market_data is not None and isinstance(market_data, (TradeData, TransactionData)):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
        elif isinstance(market_data, TickData):
            ticker = market_data.ticker
            acc_volume = market_data.total_traded_notional if use_notional else market_data.total_traded_volume

            if acc_volume is not None and np.isfinite(acc_volume) and acc_volume:
                self._accumulated_volume[ticker] = acc_volume
        elif isinstance(market_data, BarData):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
        elif market_data is not None:
            raise NotImplementedError(f'Can not handle market data type {market_data.__class__}, expect TickData, BarData, TradeData and TransactionData.')
        else:
            if ticker is not None:
                self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
            else:
                raise ValueError('Must assign market_data, or ticker and volume')

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, **kwargs):
        """
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        Parameters:
        - ticker (str): Ticker symbol for the observation.
        - value (float): Value of the observation.
        - storage (Dict[str, Dict[float, float]]): Storage dictionary for sampled observations.
        - volume_accumulated (float): Accumulated volume for the observation timestamp.

        """
        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        super().log_obs(ticker=ticker, timestamp=volume_accumulated, observation=observation, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')

        data_dict.update(
            accumulated_volume=self._accumulated_volume
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> FixedVolumeIntervalSampler:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        self.sample_storage = json_dict['sample_storage']
        self.accumulated_volume = json_dict['accumulated_volume']

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes.
        """
        super().clear()
        self._accumulated_volume.clear()


class AdaptiveVolumeIntervalSampler(FixedVolumeIntervalSampler, metaclass=abc.ABCMeta):
    """
    Concrete implementation of FixedVolumeIntervalSampler with adaptive volume interval sampling.

    Args:
    - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
    - sample_size (int): Number of samples to be stored. Default is 20.
    - baseline_window (int): Number of observations used for baseline calculation. Default is 100.

    Attributes:
    - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline.
    - sample_size (int): Number of samples to be stored.
    - baseline_window (int): Number of observations used for baseline calculation.
    - _volume_baseline (Dict[str, Union[float, Dict[str, float], Dict[str, float], Dict[str, Dict[float, float]]]]):
        Dictionary containing baseline information.

    Methods:
    - _update_volume_baseline(ticker: str, timestamp: float, volume_accumulated: float = None) -> float | None
        Updates and calculates the baseline volume for a given ticker.

    - log_obs(ticker: str, value: float, storage: Dict[str, Dict[Tuple[float, float], float]],
              volume_accumulated: float = None, timestamp: float = None, allow_oversampling: bool = False)
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

    - clear()
        Clears all stored data, including accumulated volumes and baseline information.

    Notes:
    - This class extends FixedVolumeIntervalSampler and introduces adaptive volume interval sampling.
    - The sampling_interval is a temporal interval in seconds for generating the baseline.
    - The baseline is calculated adaptively based on the provided baseline_window.

    """

    def __init__(self, sampling_interval: float = 60., sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = True):
        """
        Initialize the AdaptiveVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
        - sample_size (int): Number of samples to be stored. Default is 20.
        - baseline_window (int): Number of observations used for baseline calculation. Default is 100.
        - aligned (bool): Whether the sampling of each ticker is aligned (same temporal interval)
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size)
        self.baseline_window = baseline_window
        self.aligned_interval = aligned_interval

        self._volume_baseline = {
            'baseline': {},  # type: dict[str, float]
            'sampling_interval': {},  # type: dict[str, float]
            'obs_vol_acc_start': {},  # type: dict[str, float]
            'obs_vol_acc': {},  # type: dict[str, dict[float,float]]
        }

    def _update_volume_baseline(self, ticker: str, timestamp: float, volume_accumulated: float = None, min_obs: int = None) -> float | None:
        """
        Updates and calculates the baseline volume for a given ticker.

        Parameters:
        - ticker (str): Ticker symbol for volume baseline calculation.
        - timestamp (float): Timestamp of the observation.
        - volume_accumulated (float): Accumulated volume at the provided timestamp.

        Returns:
        - float | None: Calculated baseline volume or None if the baseline is not ready.

        """
        min_obs = self.sample_size if min_obs is None else min_obs
        volume_baseline = self._volume_baseline['baseline']
        volume_sampling_interval = self._volume_baseline['sampling_interval']

        if ticker in volume_baseline:
            return volume_baseline[ticker]

        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        if ticker in (_ := self._volume_baseline['obs_vol_acc']):
            obs_vol_acc = _[ticker]
        else:
            obs_vol_acc = _[ticker] = {}

        if not obs_vol_acc:
            obs_start_acc_vol = self._volume_baseline['obs_vol_acc_start'][ticker] = volume_accumulated  # in this case, one obs of the trade data will be missed
        else:
            obs_start_acc_vol = self._volume_baseline['obs_vol_acc_start'][ticker]

        obs_ts = (timestamp // self.sampling_interval) * self.sampling_interval
        obs_ts_start = list(obs_vol_acc)[0] if obs_vol_acc else obs_ts
        obs_ts_end = obs_ts_start + self.baseline_window * self.sampling_interval

        if timestamp < obs_ts_end:
            obs_vol_acc[obs_ts] = volume_accumulated - obs_start_acc_vol
            baseline_ready = False
        elif len(obs_vol_acc) == self.baseline_window:
            baseline_ready = True
        else:
            LOGGER.debug(f'{self.__class__.__name__} baseline validity check failed! {ticker} data missed, expect {self.baseline_window} observation, got {len(obs_vol_acc)}.')
            baseline_ready = True

        obs_vol = {}
        vol_acc_last = 0.
        for ts, vol_acc in obs_vol_acc.items():
            obs_vol[ts] = vol_acc - vol_acc_last
            vol_acc_last = vol_acc

        if len(obs_vol) < min_obs:
            baseline_est = None
        elif len(obs_vol) == 1:
            # scale the observation
            if timestamp - obs_ts > self.sampling_interval * 0.5:
                baseline_est = obs_vol[obs_ts] / (timestamp - obs_ts) * self.sampling_interval
            # can not estimate any baseline
            else:
                baseline_est = None
        else:
            if baseline_ready:
                baseline_est = np.mean([obs_vol[ts] for ts in obs_vol])
            elif timestamp - obs_ts > self.sampling_interval * 0.5:
                baseline_est = np.mean([obs_vol[ts] if ts != obs_ts else obs_vol[ts] / (timestamp - ts) * self.sampling_interval for ts in obs_vol])
            else:
                baseline_est = np.mean([obs_vol[ts] for ts in obs_vol if ts != obs_ts])

        if baseline_ready:
            if np.isfinite(baseline_est) and baseline_est > 0:
                volume_baseline[ticker] = baseline_est
            else:
                LOGGER.error(f'{ticker} Invalid estimated baseline {baseline_est}, observation window extended.')
                obs_vol_acc.clear()
                self._volume_baseline['obs_vol_acc_start'].pop(ticker)
                self._volume_baseline['sampling_interval'].pop(ticker)

        if baseline_est is not None:
            volume_sampling_interval[ticker] = baseline_est

        return baseline_est

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, allow_oversampling: bool = False, **kwargs):
        """
        Logs an observation for the given ticker at the specified volume-accumulated timestamp.

        Parameters:
        - ticker (str): Ticker symbol for the observation.
        - value (float): Value of the observation.
        - storage (Dict[str, Dict[Tuple[float, float], float]]): Storage dictionary for sampled observations.
        - volume_accumulated (float): Accumulated volume for the observation timestamp.
        - timestamp (float): Timestamp of the observation.
        - allow_oversampling (bool): Flag indicating whether oversampling is allowed.

        Raises:
        - ValueError: If timestamp is not provided.

        """
        # step 0: copy the observation
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        # step 1: calculate sampling interval
        volume_sampling_interval = self._update_volume_baseline(ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated)

        # step 2: calculate index
        if volume_sampling_interval is None:
            # baseline still in generating process, fallback to fixed temporal interval mode
            idx_ts = timestamp // self.sampling_interval
            idx_vol = 0
        elif volume_sampling_interval <= 0 or not np.isfinite(volume_sampling_interval):
            LOGGER.error(f'Invalid volume update interval for {ticker}, expect positive float, got {volume_sampling_interval}')
            return
        elif self.aligned_interval:
            volume_sampling_interval = 0.
            volume_accumulated = 0.

            volume_baseline = self._volume_baseline['baseline']
            sampling_interval = self._volume_baseline['sampling_interval']
            weights = getattr(self, 'weights', {})

            for component, vol_acc in self._accumulated_volume.items():
                weight = weights.get(component, 0.)
                vol_sampling_interval = volume_baseline.get(component, sampling_interval.get(component, 0.))

                if not weight:
                    continue

                if not vol_sampling_interval:
                    continue

                volume_accumulated += vol_acc * weight
                volume_sampling_interval += vol_sampling_interval * weight

            idx_ts = timestamp // self.sampling_interval if allow_oversampling else 0.
            idx_vol = volume_accumulated // volume_sampling_interval
        else:
            idx_ts = timestamp // self.sampling_interval if allow_oversampling else 0.
            idx_vol = volume_accumulated // volume_sampling_interval

        idx = (idx_ts, idx_vol)
        idx_vol_min = idx_vol - self.sample_size

        # step 3: update sampler
        for obs_name, obs_value in observation_copy.items():
            if obs_name not in self.sample_storage:
                raise ValueError(f'Invalid observation name {obs_name}')

            sampler = self.sample_storage[obs_name]
            storage = sampler['storage']
            mode = sampler['mode']

            if ticker in storage:
                obs_storage = storage[ticker]
            else:
                obs_storage = storage[ticker] = {}

            if idx not in obs_storage:
                obs_storage[idx] = obs_value
                self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
            else:
                if mode == 'update':
                    obs_storage[idx] = obs_value
                elif mode == 'accumulate':
                    obs_storage[idx] += obs_value
                else:
                    raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')
                self.on_entry_updated(ticker=ticker, name=obs_name, value=obs_storage[idx])

            for idx_pop in list(obs_storage):
                idx_ts, idx_vol = idx_pop

                # fallback to aligned mode
                if idx_vol == 0:
                    allow_oversampling = False
                    break
                elif idx_vol < idx_vol_min:
                    _ = obs_storage.pop(idx_pop)
                    self.on_entry_removed(ticker=ticker, name=obs_name, value=_)
                else:
                    break

            # allow max sample size of len(sampled_obs) + 1 entries. (the extra entry is the active entry)
            if not allow_oversampling and (to_pop := len(obs_storage) + 1 - self.sample_size) > 0:
                for idx_pop in list(obs_storage)[:to_pop]:
                    _ = obs_storage.pop(idx_pop)
                    self.on_entry_removed(ticker=ticker, name=obs_name, value=_)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')

        data_dict.update(
            baseline_window=self.baseline_window,
            aligned_interval=self.aligned_interval,
            volume_baseline=self._volume_baseline
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> AdaptiveVolumeIntervalSampler:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval']
        )

        self.sample_storage = json_dict['sample_storage']
        self.accumulated_volume = json_dict['accumulated_volume']
        self._volume_baseline = json_dict['volume_baseline']

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes and baseline information.
        """
        super().clear()

        self._volume_baseline['baseline'].clear()
        self._volume_baseline['sampling_interval'].clear()
        self._volume_baseline['obs_vol_acc_start'].clear()
        self._volume_baseline['obs_vol_acc'].clear()
