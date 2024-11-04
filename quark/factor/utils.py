from __future__ import annotations

import abc
import hashlib
import json
import pickle
from collections import deque
from typing import Self

import numpy as np
from algo_engine.base import MarketData, TickData, TradeData, TransactionData, OrderBook
from algo_engine.engine import MarketDataMonitor

from .sampler import *
from .. import LOGGER
from ..base import GlobalStatics
from ..base.memory_core import SharedMemoryCore

LOGGER = LOGGER.getChild('Utils')
ALPHA_05 = 0.9885  # alpha = 0.5 for each minute
ALPHA_02 = 0.9735  # alpha = 0.2 for each minute
ALPHA_01 = 0.9624  # alpha = 0.1 for each minute
ALPHA_001 = 0.9261  # alpha = 0.01 for each minute
ALPHA_0001 = 0.8913  # alpha = 0.001 for each minute


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


class FactorMonitor(MarketDataMonitor, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: str = None, meta_info: dict[str, str | float | int | bool] = None):
        """
        the init function should never accept *arg or **kwargs as parameters.
        since the from_json function depends on it.

        the __init__ function of child factor monitor:
        - must call FactorMonitor.__init__ first, since the subscription and memory core should be defined before other class initialization.
        - for the same reason, FactorMonitor.from_json, FactorMonitor.update_from_json also should be called first in child classes.
        - no *args, **kwargs like arbitrary parameter is allowed, since the from_json depends on the full signature of the method.
        - all parameters should be json serializable, as required in from_json function
        - param "subscription" is required if multiprocessing is enabled

        Args:
            name: the name of the monitor.
            monitor_id: the id of the monitor, if left unset, use uuid4().
        """
        var_names = self.__class__.__init__.__code__.co_varnames

        if 'kwargs' in var_names or 'args' in var_names:
            LOGGER.warning(f'*args and **kwargs are should not in {self.__class__.__name__} initialization. All parameters must be explicit.')

        assert name.startswith('Monitor')
        super().__init__(name=name, monitor_id=monitor_id)

        self.__additional_meta_info = {} if meta_info is None else meta_info.copy()

    def __call__(self, market_data: MarketData, allow_out_session: bool = True, **kwargs):
        # filter the out session data
        if not (GlobalStatics.PROFILE.is_market_session(market_data.timestamp) or allow_out_session):
            return

        self.on_market_data(market_data=market_data, **kwargs)

        if isinstance(market_data, TickData):
            self.on_tick_data(tick_data=market_data, **kwargs)
        elif isinstance(market_data, (TradeData, TransactionData)):
            self.on_trade_data(trade_data=market_data, **kwargs)
        elif isinstance(market_data, OrderBook):
            self.on_order_book(order_book=market_data, **kwargs)
        else:
            raise NotImplementedError(f"Can not handle market data type {type(market_data)}")

    def on_market_data(self, market_data: MarketData, **kwargs):
        pass

    def on_tick_data(self, tick_data: TickData, **kwargs) -> None:
        pass

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        pass

    def on_order_book(self, order_book: OrderBook, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def factor_names(self, subscription: list[str]) -> list[str]:
        """
        This method returns a list of string, corresponding with the keys of the what .value returns.
        This method is design to facilitate facter caching functions.
        """
        ...

    @classmethod
    def _params_grid(cls, param_range: dict[str, list[...]], param_static: dict[str, ...] = None, auto_naming: bool = None) -> list[dict[str, ...]]:
        """
        convert param grid to list of params
        Args:
            param_range: parameter range, e.g. dict(sampling_interval=[5, 15, 60], sample_size=[10, 20, 30])
            param_static: static parameter value, e.g. dict(weights=self.weights), this CAN OVERRIDE param_range

        Returns: parameter list

        """
        param_grid: list[dict] = []

        for name in param_range:
            _param_range = param_range[name]
            extended_param_grid = []

            for value in _param_range:
                if param_grid:
                    for _ in param_grid:
                        _ = _.copy()
                        _[name] = value
                        extended_param_grid.append(_)
                else:
                    extended_param_grid.append({name: value})

            param_grid.clear()
            param_grid.extend(extended_param_grid)

        if param_static:
            for _ in param_grid:
                _.update(param_static)

        if (auto_naming
                or ('name' not in param_range
                    and 'name' not in param_static
                    and auto_naming is None)):
            for i, param_dict in enumerate(param_grid):
                param_dict['name'] = f'Monitor.Grid.{cls.__name__}.{i}'

        return param_grid

    def _update_meta_info(self, meta_info: dict[str, str | float | int | bool] = None, **kwargs) -> dict[str, str | float | int | bool]:
        _meta_info = {}

        _meta_info.update(
            name=self.name,
            type=self.__class__.__name__
        )

        if meta_info:
            _meta_info.update(meta_info, **kwargs)

        if isinstance(self, FixedIntervalSampler):
            _meta_info.update(
                sampling_interval=self.sampling_interval,
                sample_size=self.sample_size
            )

        if isinstance(self, FixedVolumeIntervalSampler):
            # no additional meta info
            pass

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            _meta_info.update(
                baseline_window=self.baseline_window,
                aligned_interval=self.aligned_interval
            )

        if isinstance(self, Synthetic):
            _meta_info.update(
                index_name=self.weights.index_name
            )

        if isinstance(self, EMA):
            LOGGER.warning('Meta info for EMA may not be accurate due to precision limitation!')

            _meta_info.update(
                alpha=self.alpha,
                window=self.window
            )

        return _meta_info

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id
            # enabled=self.enabled  # the enable flag will not be serialized, this affects the handling of mask in multiprocessing
        )

        if isinstance(self, FixedIntervalSampler):
            data_dict.update(
                FixedIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, FixedVolumeIntervalSampler):
            data_dict.update(
                FixedVolumeIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            data_dict.update(
                AdaptiveVolumeIntervalSampler.to_json(self=self, fmt='dict')
            )

        if isinstance(self, Synthetic):
            data_dict.update(
                Synthetic.to_json(self=self, fmt='dict')
            )

        if isinstance(self, EMA):
            data_dict.update(
                EMA.to_json(self=self, fmt='dict')
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

        var_names = cls.__init__.__code__.co_varnames
        kwargs = {name: json_dict[name] for name in var_names if name in json_dict}

        self = cls(**kwargs)

        self.update_from_json(json_dict=json_dict)

        return self

    def to_shm(self, name: str = None, manager: SharedMemoryCore = None) -> str:
        if name is None:
            name = f'{self.monitor_id}.json'

        if manager is None:
            from ..base.memory_core import SharedMemoryCore
            manager = SharedMemoryCore()

        serialized = pickle.dumps(self)
        size = len(serialized)

        manager.init_buffer(name=name, buffer_size=size, init_value=serialized)
        # shm.close()
        return name

    @classmethod
    def from_shm(cls, name: str = None, monitor_id: str = None, manager: SharedMemoryCore = None) -> Self:
        if name is None and monitor_id is None:
            raise ValueError('Must assign a name or monitor_id.')
        if name is None:
            name = f'{monitor_id}.json'

        if manager is None:
            manager = SharedMemoryCore()

        shm = manager.get_buffer(name=name)
        self: Self = pickle.loads(bytes(shm.buffer))

        return self

    def clear(self) -> None:
        if isinstance(self, EMA):
            EMA.clear(self=self)

        if isinstance(self, Synthetic):
            Synthetic.clear(self=self)

        if isinstance(self, FixedIntervalSampler):
            FixedIntervalSampler.clear(self=self)

        if isinstance(self, FixedVolumeIntervalSampler):
            FixedVolumeIntervalSampler.clear(self=self)

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            AdaptiveVolumeIntervalSampler.clear(self=self)

    def update_from_json(self, json_dict: dict) -> Self:
        """
        a utility function for .from_json()

        Note that calling this function DOES NOT CLEAR the object
        This function will add /update / override the data from json dict

        Call .clear() explicitly if a re-construct is needed.
        """

        if isinstance(self, EMA):
            self: EMA
            for name in json_dict['ema']:
                self.register_ema(name=name)

                self._ema_memory[name].update(json_dict['ema_memory'][name])
                self._ema_current[name].update(json_dict['ema_current'][name])
                self.ema[name].update(json_dict['ema'][name])

        if isinstance(self, Synthetic):
            self: Synthetic

            self.base_price.update(json_dict['base_price'])
            self.last_price.update(json_dict['last_price'])
            self.synthetic_base_price = json_dict['synthetic_base_price']

        if isinstance(self, FixedIntervalSampler):
            self: FixedIntervalSampler

            for topic, sampler_json in json_dict['sampler_data'].items():
                sampler_data = SamplerData.from_json(sampler_json)
                self.sampler_data[topic] = sampler_data

        if isinstance(self, FixedVolumeIntervalSampler):
            self: FixedVolumeIntervalSampler

            self._accumulated_volume.update(json_dict['accumulated_volume'])

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            self: AdaptiveVolumeIntervalSampler

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

    def _param_range(self) -> dict[str, list[...]]:
        # give some simple parameter range
        params_range = {}
        if isinstance(self, FixedIntervalSampler):
            params_range.update(
                sampling_interval=[5, 15, 60],
                sample_size=[10, 20, 30]
            )

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            params_range.update(
                aligned_interval=[True, False]
            )

        if isinstance(self, EMA):
            params_range.update(
                alpha=[ALPHA_05, ALPHA_02, ALPHA_01, ALPHA_001, ALPHA_0001]
            )

        return params_range

    def _param_static(self) -> dict[str, ...]:
        param_static = {}

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            param_static.update(
                baseline_window=self.baseline_window
            )

        if isinstance(self, Synthetic):
            param_static.update(
                weights=self.weights
            )

        return param_static

    def params_list(self) -> list[dict[str, ...]]:
        """
        This method is designed to facilitate the grid cv process.
        The method will return a list of params to initialize monitor.
        e.g.
        > params_list = self.params_grid()
        > monitors = [SomeMonitor(**_) for _ in params_list]
        > # cross validation process ...
        Returns: a list of dict[str, ...]

        """

        params_list = self._params_grid(
            param_range=self._param_range(),
            param_static=self._param_static()
        )

        return params_list

    @property
    def params(self) -> dict:
        var_names = self.__class__.__init__.__code__.co_varnames
        params = dict()

        for var_name in var_names:
            if var_name in ['self', 'args', 'kwargs', 'monitor_id']:
                continue
            var_value = getattr(self, var_name)
            params[var_name] = var_value
        return params

    @property
    def serializable(self) -> bool:
        return True

    @property
    def use_shm(self) -> bool:
        return False

    @property
    def meta(self) -> dict[str, str | float | int | bool]:
        meta_info = self._update_meta_info(meta_info=self.__additional_meta_info)
        return {k: meta_info[k] for k in sorted(meta_info)}

    def digest(self, encoding: str = 'utf-8') -> str:
        hashed_str = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode(encoding=encoding)).hexdigest()
        return hashed_str


class EMA(object):
    """
    Use EMA module with samplers to get best results
    """

    def __init__(self, alpha: float = None, window: int = None):
        self.alpha = alpha if alpha else 1 - 2 / (window + 1)
        self.window = window if window else round(2 / (1 - alpha) - 1)

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
        if name in self.ema:
            LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
            return self.ema[name]

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

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            alpha=self.alpha,
            window=self.window,
            ema_memory=self._ema_memory,
            ema_current=self._ema_current,
            ema=self.ema
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
            alpha=json_dict['alpha'],
            window=json_dict['window']
        )

        for name in json_dict['ema']:
            self.register_ema(name=name)

            self._ema_memory[name].update(json_dict['ema_memory'][name])
            self._ema_current[name].update(json_dict['ema_current'][name])
            self.ema[name].update(json_dict['ema'][name])

        return self

    def clear(self):
        self._ema_memory.clear()
        self._ema_current.clear()
        self.ema.clear()


class Synthetic(object, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float]):
        self.weights: IndexWeight = weights if isinstance(weights, IndexWeight) else IndexWeight(index_name='synthetic', **weights)
        self.weights.normalize()

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price: float = 1.

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        return self.weights.composite(values=values, replace_na=replace_na)

    def update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        base_price = self.base_price.get(ticker, np.nan)
        if not np.isfinite(base_price):
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

        self = cls(
            weights=IndexWeight(index_name=json_dict['index_name'], **json_dict['weights'])
        )

        self.base_price.update(json_dict['base_price'])
        self.last_price.update(json_dict['last_price'])
        self.synthetic_base_price = json_dict['synthetic_base_price']

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

            assert weight > 0, f'Weight of {ticker} in {self.weights.index_name} must be greater than zero.'
            weight_list.append(weight)

            if np.isfinite(last_price) and np.isfinite(base_price):
                price_list.append(last_price / base_price)
            else:
                price_list.append(1.)

        if sum(weight_list):
            synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        else:
            synthetic_index = 1.

        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(self.last_price)


__all__ = ['FactorMonitor',
           'EMA', 'ALPHA_05', 'ALPHA_02', 'ALPHA_01', 'ALPHA_001', 'ALPHA_0001',
           'Synthetic', 'IndexWeight']
