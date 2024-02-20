from __future__ import annotations

import abc
import enum
import json
from collections import deque
from typing import Self

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData, TickData, BarData

from .memory_core import SyncMemoryCore
from .. import LOGGER
from ..Calibration.dummies import is_market_session

__all__ = ['IndexWeight', 'EMA', 'Synthetic', 'SamplerMode', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler']


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
    """
    Use EMA module with samplers to get best results
    """

    def __init__(self, alpha: float = None, window: int = None, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        self.alpha = alpha if alpha else 1 - 2 / (window + 1)
        self.window = window if window else round(2 / (1 - alpha) - 1)

        self.subscription: set[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.__class__.__name__} should have an alpha between 0 to 1')

        self._ema_memory: dict[str, dict[str, float]] = getattr(self, '_ema_memory', {})
        self._ema_current: dict[str, dict[str, float]] = getattr(self, '_ema_current', {})
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

    def register_ema(self, name: str) -> dict[str, float]:
        self._ema_memory[name] = {}
        self._ema_current[name] = {}
        self.ema[name] = {}

        return self.ema[name]

    def update_ema(self, ticker: str, replace_na: float = np.nan, **update_data: float):
        """
        update ema on call

        Args:
            ticker: the ticker of the
            replace_na: replace the memory value with the gaven if it is nan
            **update_data: {'ema_a': 1, 'ema_b': 2}

        Returns: None

        """
        # update the current
        for entry_name, value in update_data.items():
            if entry_name not in self._ema_current:
                LOGGER.warning(f'Entry {entry_name} not registered')
                continue

            if not np.isfinite(value):
                LOGGER.warning(f'Value for {entry_name} not valid, expect float, got {value}, ignored to prevent data-contamination.')
                continue

            current = self._ema_current[entry_name][ticker] = value
            memory = self._ema_memory[entry_name].get(ticker, np.nan)

            if np.isfinite(memory):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)
            else:
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)

    def accumulate_ema(self, ticker: str, replace_na: float = np.nan, **accumulative_data: float):
        # add to current
        for entry_name, value in accumulative_data.items():
            if entry_name not in self._ema_current:
                LOGGER.warning(f'Entry {entry_name} not registered')
                continue

            if not np.isfinite(value):
                LOGGER.warning(f'Value for {entry_name} not valid, expect float, got {value}, ignored to prevent data-contamination.')
                continue

            current = self._ema_current[entry_name][ticker] = self._ema_current[entry_name].get(ticker, 0.) + value
            memory = self._ema_memory[entry_name].get(ticker, np.nan)

            if np.isfinite(memory):
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=memory, alpha=self.alpha)
            else:
                self.ema[entry_name][ticker] = self.calculate_ema(value=current, memory=replace_na, alpha=self.alpha)

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            alpha=self.alpha,
            window=self.window,
            ema_memory=self._ema_memory,
            ema_current=self._ema_current,
            ema=self.ema
        )

        if with_subscription:
            data_dict['subscription'] = list(self.subscription)

        if with_memory_core:
            data_dict['memory_core'] = self.memory_core.to_json(fmt='dict')

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

        kwargs = {}

        if 'subscription' in json_dict:
            kwargs['subscription'] = json_dict('subscription')

        if 'memory_core' in json_dict:
            kwargs['memory_core'] = SyncMemoryCore.from_json(json_dict['memory_core'])

        self = cls(
            alpha=json_dict['alpha'],
            window=json_dict['window'],
            **kwargs
        )

        self._ema_memory.update(json_dict['ema_memory'])
        self._ema_current.update(json_dict['ema_current'])
        self.ema.update(json_dict['ema'])

        return self

    def clear(self):
        for storage in self._ema_memory.values():
            storage.clear()

        for storage in self._ema_current.values():
            storage.clear()

        for storage in self.ema.values():
            storage.clear()

        self._ema_memory.clear()
        self._ema_current.clear()
        self.ema.clear()


class Synthetic(object, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float], subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        self.weights: IndexWeight = weights if isinstance(weights, IndexWeight) else IndexWeight(index_name='synthetic', **weights)
        self.weights.normalize()

        self.subscription: set[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price: float = 1.

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        return self.weights.composite(values=values, replace_na=replace_na)

    def update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        if ticker not in self.base_price:
            self.base_price[ticker] = market_price

        self.last_price[ticker] = market_price

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.weights.index_name,
            weights=dict(self.weights),
            base_price=self.base_price,
            last_price=self.last_price,
            synthetic_base_price=self.synthetic_base_price
        )

        if with_subscription:
            data_dict['subscription'] = list(self.subscription)

        if with_memory_core:
            data_dict['memory_core'] = self.memory_core.to_json(fmt='dict')

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

        kwargs = {}

        if 'subscription' in json_dict:
            kwargs['subscription'] = json_dict('subscription')

        if 'memory_core' in json_dict:
            kwargs['memory_core'] = SyncMemoryCore.from_json(json_dict['memory_core'])

        self = cls(
            weights=IndexWeight(index_name=json_dict['index_name'], **json_dict['weights']),
            **kwargs
        )

        self.base_price.update(json_dict['base_price'])
        self.last_price.update(json_dict['last_price'])
        self.synthetic_base_price.value = json_dict['synthetic_base_price']

        return self

    def clear(self):
        self.base_price.clear()
        self.last_price.clear()
        self.synthetic_base_price = 1.

    @property
    def synthetic_index(self):
        price_list = []
        weight_list = []

        for ticker, weight in self.weights.items():
            last_price = self.last_price.get(ticker, np.nan)
            base_price = self.base_price.get(ticker, np.nan)

            if np.isfinite(last_price) and np.isfinite(base_price) and weight:
                weight_list.append(self.weights[ticker])
                price_list.append(last_price / base_price)
            else:
                weight_list.append(0.)
                price_list.append(1.)

        synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(self.last_price)


class SamplerMode(enum.Enum):
    update = 'update'
    accumulate = 'accumulate'


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

    def __init__(self, sampling_interval: float = 1., sample_size: int = 60, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        """
        Initialize the FixedIntervalSampler.

        Parameters:
        - sampling_interval (float): Time interval between consecutive samples (in seconds). Default is 1.
        - sample_size (int): Number of samples to be stored. Default is 60.
        """
        self.sampling_interval = sampling_interval
        self.sample_size = sample_size

        self.subscription: set[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        self.sample_storage = getattr(self, 'sample_storage', {})  # to avoid de-reference the dict using nested inheritance

        # Warning for sampling_interval
        if sampling_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive sampling_interval')

        # Warning for sample_interval by Shannon's Theorem
        if sample_size <= 2:
            LOGGER.warning(f"{self.__class__.__name__} should have a larger sample_size, by Shannon's Theorem, sample_size should be greater than 2")

    def register_sampler(self, name: str, mode: str | SamplerMode = 'update'):
        if name in self.sample_storage:
            raise ValueError(f'name {name} already registered in {self.__class__.__name__}!')

        if isinstance(mode, SamplerMode):
            mode = mode.value

        if mode not in ['update', 'accumulate']:
            raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

        sample_storage = self.sample_storage[name] = dict(
            storage={},
            index={},
            mode=mode
        )

        return sample_storage

    def get_sampler(self, name: str) -> dict[str, deque]:
        if name not in self.sample_storage:
            raise ValueError(f'name {name} not found in {self.__class__.__name__}!')

        return self.sample_storage[name]['storage']

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        observation_copy = {}

        if observation is not None:
            observation_copy.update(observation)

        observation_copy.update(kwargs)

        idx = timestamp // self.sampling_interval

        for obs_name, obs_value in observation_copy.items():
            if obs_name not in self.sample_storage:
                raise ValueError(f'Invalid observation name {obs_name}')

            sampler = self.sample_storage[obs_name]
            storage: dict[str, deque] = sampler['storage']
            indices: dict = sampler['index']
            mode = sampler['mode']

            if ticker in storage:
                obs_storage = storage[ticker]
            else:
                obs_storage = storage[ticker] = deque(maxlen=self.sample_size)

            last_idx = indices.get(ticker, 0)

            if idx > last_idx:
                obs_storage.append(obs_value)
                indices[ticker] = idx
                self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
            else:
                if mode == 'update':
                    last_obs = obs_storage[-1] = obs_value
                elif mode == 'accumulate':
                    last_obs = obs_storage[-1] = obs_value + obs_storage[-1]
                else:
                    raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

                self.on_entry_updated(ticker=ticker, name=obs_name, value=last_obs)

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

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            sampling_interval=self.sampling_interval,
            sample_size=self.sample_size,
            sample_storage={name: dict(storage={ticker: list(dq) for ticker, dq in value['storage'].items()},
                                       index=value['index'],
                                       mode=value['mode'])
                            for name, value in self.sample_storage.items()}
        )

        if with_subscription:
            data_dict['subscription'] = list(self.subscription)

        if with_memory_core:
            data_dict['memory_core'] = self.memory_core.to_json(fmt='dict')

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

        kwargs = {}

        if 'subscription' in json_dict:
            kwargs['subscription'] = json_dict('subscription')

        if 'memory_core' in json_dict:
            kwargs['memory_core'] = SyncMemoryCore.from_json(json_dict['memory_core'])

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            **kwargs
        )

        self.sample_storage.update({name: dict(storage={ticker: deque(data, maxlen=self.sample_size) for ticker, data in value['storage'].items()},
                                               index=value['index'],
                                               mode=value['mode'])
                                    for name, value in json_dict['sample_storage'].items()})

        return self

    def clear(self):
        """
        Clears all stored data.
        """
        for name, sample_storage in self.sample_storage.items():
            for ticker, dq in sample_storage['storage'].items():
                dq.clear()

            self.sample_storage[name]['index'].clear()

        # using this code will require the sampler to be registered again.
        self.sample_storage.clear()

    def loc_obs(self, name: str, ticker: str, index: int | slice = None) -> float | list[float]:
        sampler = self.get_sampler(name=name)
        observation = sampler.get(ticker, [])

        if index is None:
            return list(observation)
        else:
            return list(observation)[index]

    def active_obs(self, name: str) -> dict[str, float]:
        sampler = self.get_sampler(name=name)
        last_obs = {}

        for ticker, observation in sampler.items():
            if observation:
                last_obs[ticker] = observation[-1]

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

    def __init__(self, sampling_interval: float = 100., sample_size: int = 20, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        """
        Initialize the FixedVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Volume interval between consecutive samples. Default is 100.
        - sample_size (int): Number of samples to be stored. Default is 20.
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, subscription=subscription, memory_core=memory_core)
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
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        kwargs = {}

        if 'subscription' in json_dict:
            kwargs['subscription'] = json_dict('subscription')

        if 'memory_core' in json_dict:
            kwargs['memory_core'] = SyncMemoryCore.from_json(json_dict['memory_core'])

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            **kwargs
        )

        self.sample_storage.update(json_dict['sample_storage'])
        self._accumulated_volume.update(json_dict['accumulated_volume'])

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

    def __init__(self, sampling_interval: float = 60., sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = True, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        """
        Initialize the AdaptiveVolumeIntervalSampler.

        Parameters:
        - sampling_interval (float): Temporal interval between consecutive samples for generating the baseline. Default is 60.
        - sample_size (int): Number of samples to be stored. Default is 20.
        - baseline_window (int): Number of observations used for baseline calculation. Default is 100.
        - aligned (bool): Whether the sampling of each ticker is aligned (same temporal interval)
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, subscription=subscription, memory_core=memory_core)
        self.baseline_window = baseline_window
        self.aligned_interval = aligned_interval

        self._volume_baseline = {
            'baseline': {},  # type: dict[str, float]
            'sampling_interval': {},  # type: dict[str, float]
            'obs_vol_acc_start': {},  # type: dict[str, float]
            'obs_index': {},  # type: dict[str, dict[float,float]]
            'obs_vol_acc': {},  # type: dict[str, deque]
        }

    def register_sampler(self, name: str, mode='update'):
        sample_storage = super().register_sampler(name=name, mode=mode)

        sample_storage['index_vol'] = {}

        return sample_storage

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

        volume_baseline_dict: dict[str, float] = self._volume_baseline['baseline']
        obs_vol_acc_start_dict: dict[str, float] = self._volume_baseline['obs_vol_acc_start']
        obs_index_dict: dict[str, float] = self._volume_baseline['obs_index']

        volume_baseline = volume_baseline_dict.get(ticker, np.nan)
        volume_accumulated = self._accumulated_volume.get(ticker, 0.) if volume_accumulated is None else volume_accumulated

        if np.isfinite(volume_baseline):
            return volume_baseline

        if ticker in (_ := self._volume_baseline['obs_vol_acc']):
            obs_vol_acc: deque = _[ticker]
        else:
            obs_vol_acc = _[ticker] = deque(maxlen=self.baseline_window)

        if not obs_vol_acc:
            # in this case, one obs of the trade data will be missed
            obs_start_acc_vol = obs_vol_acc_start_dict[ticker] = volume_accumulated
        else:
            obs_start_acc_vol = obs_vol_acc_start_dict[ticker]

        last_idx = obs_index_dict.get(ticker, 0)
        obs_idx = obs_index_dict[ticker] = timestamp // self.sampling_interval
        obs_ts = obs_idx * self.sampling_interval
        volume_acc = volume_accumulated - obs_start_acc_vol
        baseline_ready = False

        if not obs_vol_acc:
            obs_vol_acc.append(volume_acc)
        elif obs_idx == last_idx:
            obs_vol_acc[-1] = volume_acc
        # in this case, the obs_vol_acc is full, and a new index received, baseline is ready
        elif len(obs_vol_acc) == self.baseline_window:
            baseline_ready = True
        else:
            obs_vol_acc.append(volume_acc)

        # convert vol_acc to vol
        obs_vol = []
        vol_acc_last = 0.
        for vol_acc in obs_vol_acc:
            obs_vol.append(vol_acc - vol_acc_last)
            vol_acc_last = vol_acc

        if len(obs_vol) < min_obs:
            baseline_est = None
        else:
            if baseline_ready:
                baseline_est = np.mean(obs_vol)
            else:
                obs_vol_history = obs_vol[:-1]
                obs_vol_current_adjusted = obs_vol[-1] / max(1., timestamp - obs_ts) * self.sampling_interval

                if timestamp - obs_ts > self.sampling_interval * 0.5 and obs_vol_current_adjusted is not None:
                    obs_vol_history.append(obs_vol_current_adjusted)

                baseline_est = np.mean(obs_vol_history) if obs_vol_history else None

        if baseline_ready:
            if np.isfinite(baseline_est) and baseline_est > 0:
                self._volume_baseline['baseline'][ticker] = baseline_est
            else:
                LOGGER.error(f'{ticker} Invalid estimated baseline {baseline_est}, observation window extended.')
                obs_vol_acc.clear()
                self._volume_baseline['obs_vol_acc_start'].pop(ticker)
                self._volume_baseline['sampling_interval'].pop(ticker)

        if baseline_est is not None:
            self._volume_baseline['sampling_interval'][ticker] = baseline_est

        return baseline_est

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, **kwargs):
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

            idx_ts = 0.
            idx_vol = volume_accumulated // volume_sampling_interval
        else:
            idx_ts = 0.
            idx_vol = volume_accumulated // volume_sampling_interval

        # step 3: update sampler
        for obs_name, obs_value in observation_copy.items():
            if obs_name not in self.sample_storage:
                raise ValueError(f'Invalid observation name {obs_name}')

            sampler = self.sample_storage[obs_name]
            storage: dict[str, deque] = sampler['storage']
            indices_timestamp: dict[str, float] = sampler['index']
            indices_volume: dict[str, float] = sampler['index_vol']
            mode = sampler['mode']

            if ticker in storage:
                obs_storage = storage[ticker]
            else:
                obs_storage = storage[ticker] = deque(maxlen=self.sample_size)

            last_idx_ts = indices_timestamp.get(ticker, 0)
            last_idx_vol = indices_volume.get(ticker, 0)

            if idx_ts > last_idx_ts or idx_vol > last_idx_vol:
                obs_storage.append(obs_value)
                indices_timestamp[ticker] = idx_ts
                indices_volume[ticker] = idx_vol
                self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
            else:
                if mode == 'update':
                    last_obs = obs_storage[-1] = obs_value
                elif mode == 'accumulate':
                    last_obs = obs_storage[-1] = obs_value + obs_storage[-1]
                else:
                    raise NotImplementedError(f'Invalid mode {mode}, expect "update" or "accumulate".')

                self.on_entry_updated(ticker=ticker, name=obs_name, value=last_obs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict: dict = super().to_json(fmt='dict')

        for name, sample_storage in self.sample_storage.items():
            data_dict['sample_storage'][name]['index_vol'] = dict(sample_storage['index_vol'].items())

        data_dict.update(
            baseline_window=self.baseline_window,
            aligned_interval=self.aligned_interval,
            volume_baseline=dict(
                baseline=dict(self._volume_baseline['baseline']),
                sampling_interval=dict(self._volume_baseline['sampling_interval']),
                obs_vol_acc_start=dict(self._volume_baseline['obs_vol_acc_start']),
                obs_index=dict(self._volume_baseline['obs_index']),
                obs_vol_acc={ticker: list(dq) for ticker, dq in self._volume_baseline['obs_vol_acc'].items()}
            )
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

        kwargs = {}

        if 'subscription' in json_dict:
            kwargs['subscription'] = json_dict('subscription')

        if 'memory_core' in json_dict:
            kwargs['memory_core'] = SyncMemoryCore.from_json(json_dict['memory_core'])

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            **kwargs
        )

        self.sample_storage.update(
            {name: dict(storage={ticker: deque(data, maxlen=self.sample_size) for ticker, data in value['storage'].items()},
                        index=value['index'],
                        index_vol=value['index_vol'],
                        mode=value['mode'])
             for name, value in json_dict['sample_storage'].items()}
        )

        self._accumulated_volume.update(json_dict['accumulated_volume'])

        self._volume_baseline['baseline'].update(json_dict['volume_baseline']['baseline'])
        self._volume_baseline['sampling_interval'].update(json_dict['volume_baseline']['sampling_interval'])
        self._volume_baseline['obs_vol_acc_start'].update(json_dict['volume_baseline']['obs_vol_acc_start'])
        self._volume_baseline['obs_index'].update(json_dict['volume_baseline']['obs_index'])
        for ticker, data in json_dict['volume_baseline']['obs_vol_acc'].items():
            if ticker in self._volume_baseline['obs_vol_acc']:
                self._volume_baseline['obs_vol_acc'][ticker].extend(data)
            else:
                self._volume_baseline['obs_vol_acc'][ticker] = deque(data, maxlen=self.baseline_window)

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes and baseline information.
        """
        super().clear()

        self._volume_baseline['baseline'].clear()
        self._volume_baseline['sampling_interval'].clear()
        self._volume_baseline['obs_vol_acc_start'].clear()
        self._volume_baseline['obs_index'].clear()
        self._volume_baseline['obs_vol_acc'].clear()
