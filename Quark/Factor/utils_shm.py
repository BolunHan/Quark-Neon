from __future__ import annotations

import abc
import json
from typing import Self

import numpy as np
from PyQuantKit import MarketData

from .memory_core import SyncMemoryCore, NamedVector, Deque, FloatValue
from .utils import IndexWeight, SamplerMode, EMA as _EMA, Synthetic as _Synthetic, FixedIntervalSampler as _FIS, FixedVolumeIntervalSampler as _FVS, AdaptiveVolumeIntervalSampler as _AVS
from .. import LOGGER

__all__ = ['EMA', 'Synthetic', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler']


class EMA(_EMA, metaclass=abc.ABCMeta):
    def __init__(self, alpha: float = None, window: int = None, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(alpha=alpha, window=window, subscription=subscription, memory_core=memory_core)
        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self._ema_memory: dict[str, NamedVector] = getattr(self, '_ema_memory', {})
        self._ema_current: dict[str, NamedVector] = getattr(self, '_ema_current', {})
        self.ema: dict[str, NamedVector] = getattr(self, 'ema', {})

    def register_ema(self, name: str):
        self._ema_memory[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.memory', dtype='NamedVector')
        self._ema_current[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.current', dtype='NamedVector')
        self.ema[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.value', dtype='NamedVector')

        return self.ema[name]

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            alpha=self.alpha,
            window=self.window,
            ema_memory={entry_name: dict(storage) for entry_name, storage in self._ema_memory.items()},
            ema_current={entry_name: dict(storage) for entry_name, storage in self._ema_current.items()},
            ema={entry_name: dict(storage) for entry_name, storage in self.ema.items()}
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

    def clear(self):
        for storage in self._ema_memory.values():
            storage.unlink()

        for storage in self._ema_current.values():
            storage.unlink()

        for storage in self.ema.values():
            storage.unlink()

        self._ema_memory.clear()
        self._ema_current.clear()
        self.ema.clear()


class Synthetic(_Synthetic, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float], subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(weights=weights, subscription=subscription, memory_core=memory_core)

        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self.base_price: NamedVector = self.memory_core.register(name='base_price', dtype='NamedVector')
        self.last_price: NamedVector = self.memory_core.register(name='last_price', dtype='NamedVector')
        self.synthetic_base_price: FloatValue = self.memory_core.register(name='synthetic_base_price', dtype='FloatValue', size=8, value=1.)

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            index_name=self.weights.index_name,
            weights=dict(self.weights),
            base_price=dict(self.base_price),
            last_price=dict(self.last_price),
            synthetic_base_price=self.synthetic_base_price.value
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
        self.base_price.clear(silent=True)
        self.last_price.clear(silent=True)
        self.synthetic_base_price.value = 1

        # send cleared NamedVector to shm
        self.base_price.to_shm()
        self.last_price.to_shm()

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

        if sum(weight_list):
            synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price.value
        else:
            synthetic_index = 1.

        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(dict(self.last_price))


class FixedIntervalSampler(_FIS, metaclass=abc.ABCMeta):

    def __init__(self, sampling_interval: float = 1., sample_size: int = 60, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, subscription=subscription, memory_core=memory_core)

        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self.sample_storage = getattr(self, 'sample_storage', {})  # to avoid de-reference the dict using nested inheritance

    def register_sampler(self, name: str, mode: str | SamplerMode = 'update'):
        sample_storage = super().register_sampler(name=name, mode=mode)
        sample_storage['index'] = self.memory_core.register(name=f'Sampler.{name}.index', dtype='NamedVector')

        return sample_storage

    def on_subscription(self, subscription: list[str] = None):
        if subscription:
            subscription = set(subscription) | set(self.subscription) if self.subscription else set()
            self.subscription.clear()
            self.subscription.extend(subscription)

        for name, sample_storage in self.sample_storage.items():
            for ticker in self.subscription:
                if ticker in sample_storage['index']:
                    continue

                dq: Deque = self.memory_core.register(name=f'Sampler.{name}.{ticker}', dtype='Deque', maxlen=self.sample_size)
                sample_storage['storage'][ticker] = dq
                sample_storage['index'][ticker] = 0

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, **kwargs):
        return super().log_obs(ticker=ticker, timestamp=timestamp, observation=observation, auto_register=False, **kwargs)

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = dict(
            sampling_interval=self.sampling_interval,
            sample_size=self.sample_size,
            sample_storage={name: dict(storage={ticker: list(dq) for ticker, dq in value['storage'].items()},
                                       index=dict(value['index']),
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

        for name, sampler in json_dict['sample_storage'].items():
            mode = sampler['mode']
            new_sampler = self.register_sampler(name=name, mode=mode)
            new_sampler['index'].update(sampler['index'])

            for ticker, data in sampler['storage'].items():
                if ticker in new_sampler:
                    new_sampler['storage'][ticker].extend(data)
                else:
                    LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')

        return self

    def clear(self):
        """
        Clears all stored data.
        """
        for name, sample_storage in self.sample_storage.items():
            for ticker, dq in sample_storage['storage'].items():
                dq.unlink()

            self.sample_storage[name]['index'].unlink()

        # using this code will require the sampler to be registered again.
        self.sample_storage.clear()


class FixedVolumeIntervalSampler(FixedIntervalSampler, _FVS, metaclass=abc.ABCMeta):
    def __init__(self, sampling_interval: float = 100., sample_size: int = 20, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, subscription=subscription, memory_core=memory_core)
        self._accumulated_volume: NamedVector = self.memory_core.register(name=f'Sampler.Volume.Accumulated', dtype='NamedVector')

    def on_subscription(self, subscription: list[str] = None):
        _FVS.on_subscription(self=self, subscription=subscription)

    def accumulate_volume(self, ticker: str = None, volume: float = 0., market_data: MarketData = None, use_notional: bool = False):
        return _FVS.accumulate_volume(self=self, ticker=ticker, volume=volume, market_data=market_data, use_notional=use_notional)

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, **kwargs):
        return _FVS.log_obs(self=self, ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated, observation=observation, auto_register=False, **kwargs)

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', with_subscription=with_subscription, with_memory_core=with_memory_core, )

        data_dict.update(
            accumulated_volume=dict(self._accumulated_volume.items())
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

        for name, sampler in json_dict['sample_storage'].items():
            mode = sampler['mode']
            new_sampler = self.register_sampler(name=name, mode=mode)
            new_sampler['index'].update(sampler['index'])

            for ticker, data in sampler['storage'].items():
                if ticker in new_sampler:
                    new_sampler['storage'][ticker].extend(data)
                else:
                    LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')

        self._accumulated_volume.update(json_dict['accumulated_volume'])

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes.
        """
        super().clear()

        self._accumulated_volume.clear(silent=True)

        # send cleared NamedVector to shm
        self._accumulated_volume.to_shm()


class AdaptiveVolumeIntervalSampler(FixedVolumeIntervalSampler, _AVS, metaclass=abc.ABCMeta):

    def __init__(self, sampling_interval: float = 60., sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = True, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, subscription=subscription, memory_core=memory_core)
        self.baseline_window = baseline_window
        self.aligned_interval = aligned_interval

        self._volume_baseline = {
            'baseline': self.memory_core.register(name='sampler.baseline', dtype='NamedVector'),  # type: dict[str, float]
            'sampling_interval': self.memory_core.register(name='sampler.sampling_interval', dtype='NamedVector'),  # type: dict[str, float]
            'obs_vol_acc_start': self.memory_core.register(name='sampler.obs_vol_acc_start', dtype='NamedVector'),  # type: dict[str, float]
            'obs_index': self.memory_core.register(name='sampler.obs_index', dtype='NamedVector'),  # type: dict[str, dict[float,float]]
            'obs_vol_acc': {},  # type: dict[str, Deque]
        }

    def on_subscription(self, subscription: list[str] = None):
        super().on_subscription(subscription=subscription)

        for name, sample_storage in self.sample_storage.items():
            for ticker in self.subscription:
                if ticker in sample_storage['index_vol']:
                    continue

                sample_storage['index_vol'][ticker] = 0.

        for ticker in self.subscription:
            if ticker in self._volume_baseline['obs_index']:
                continue

            self._volume_baseline['baseline'][ticker] = np.nan
            self._volume_baseline['sampling_interval'][ticker] = np.nan
            self._volume_baseline['obs_vol_acc_start'][ticker] = np.nan
            self._volume_baseline['obs_index'][ticker] = 0.
            self._volume_baseline['obs_vol_acc'][ticker] = self.memory_core.register(name=f'sampler.obs_vol_acc.{ticker}', dtype='Deque', maxlen=self.baseline_window)

    def register_sampler(self, name: str, mode='update'):
        sample_storage = super().register_sampler(name=name, mode=mode)

        sample_storage['index_vol'] = self.memory_core.register(name=f'Sampler.{name}.index_vol', dtype='NamedVector')

        return sample_storage

    def _update_volume_baseline(self, ticker: str, timestamp: float, volume_accumulated: float = None, min_obs: int = None, auto_register=False) -> float | None:
        return _AVS._update_volume_baseline(self=self, ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated, min_obs=min_obs, auto_register=False)

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, **kwargs):
        return _AVS.log_obs(self=self, ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated, observation=observation, auto_register=False, **kwargs)

    def to_json(self, fmt='str', with_subscription: bool = False, with_memory_core: bool = False, **kwargs) -> str | dict:
        data_dict: dict = super().to_json(fmt='dict', with_subscription=with_subscription, with_memory_core=with_memory_core)

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

        for name, sampler in json_dict['sample_storage'].items():
            mode = sampler['mode']
            new_sampler = self.register_sampler(name=name, mode=mode)
            new_sampler['index'].update(sampler['index'])
            new_sampler['index_vol'].update(sampler['index_vol'])

            for ticker, data in sampler['storage'].items():
                if ticker in new_sampler:
                    new_sampler['storage'][ticker].extend(data)
                else:
                    LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')

        self._accumulated_volume.update(json_dict['accumulated_volume'])

        self._volume_baseline['baseline'].update(json_dict['volume_baseline']['baseline'])
        self._volume_baseline['sampling_interval'].update(json_dict['volume_baseline']['sampling_interval'])
        self._volume_baseline['obs_vol_acc_start'].update(json_dict['volume_baseline']['obs_vol_acc_start'])
        self._volume_baseline['obs_index'].update(json_dict['volume_baseline']['obs_index'])

        for ticker, data in json_dict['volume_baseline']['obs_vol_acc'].items():
            if ticker in self._volume_baseline['obs_vol_acc']:
                self._volume_baseline['obs_vol_acc'][ticker].extend(data)
            else:
                LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')

        return self

    def clear(self):
        """
        Clears all stored data, including accumulated volumes and baseline information.
        """
        # can not call super clear since the FIS would clear the sample_storage before unlink
        # super().clear()

        for name, sample_storage in self.sample_storage.items():
            for ticker, dq in sample_storage['storage'].items():
                dq.unlink()

            self.sample_storage[name]['index'].unlink()
            self.sample_storage[name]['index_vol'].unlink()

        self.sample_storage.clear()

        self._accumulated_volume.clear(silent=True)
        self._volume_baseline['baseline'].clear(silent=True)
        self._volume_baseline['sampling_interval'].clear(silent=True)
        self._volume_baseline['obs_vol_acc_start'].clear(silent=True)
        self._volume_baseline['obs_index'].clear(silent=True)
        for ticker in self._accumulated_volume:
            self._volume_baseline['obs_vol_acc'][ticker].unlink()
        self._volume_baseline['obs_vol_acc'].clear()

        # send cleared NamedVector to shm
        self._accumulated_volume.to_shm()
        self._volume_baseline['baseline'].to_shm()
        self._volume_baseline['sampling_interval'].to_shm()
        self._volume_baseline['obs_vol_acc_start'].to_shm()
        self._volume_baseline['obs_index'].to_shm()
