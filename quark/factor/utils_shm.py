from __future__ import annotations

import abc
import json
import pickle
import time
import traceback
from typing import Self

import numpy as np
import pandas as pd
from algo_engine.base import MarketData

from . import collect_factor, N_CORES
from .utils import FactorMonitor as _FM, ConcurrentMonitorManager as _CMM, IndexWeight, SamplerMode, EMA as _EMA, Synthetic as _Synthetic, FixedIntervalSampler as _FIS, FixedVolumeIntervalSampler as _FVS, AdaptiveVolumeIntervalSampler as _AVS
from .. import LOGGER
from ..base.memory_core import SharedMemoryCore, SyncMemoryCore, NamedVector, Deque, FloatValue

LOGGER = LOGGER.getChild('Utils.SHM')
__all__ = ['EMA', 'Synthetic', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler']


class FactorMonitor(_FM, metaclass=abc.ABCMeta):
    """
    a variation of the FactorMonitor, but using shared memory technique to support multiprocessing.
    - "on_subscription" should be called AFTER the __init__ function, so that shm memory can be properly initialized.
    - "on_subscription" should be called before added to MDS, so that shm memory can be properly initialized.
    """

    def __init__(self, name: str, subscription: list[str], monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

        self.subscription: list[str] = subscription
        self.memory_core: SyncMemoryCore = SyncMemoryCore(prefix=self.monitor_id, dummy=False if N_CORES > 1 else True)

    def __call__(self, market_data: MarketData, allow_out_session: bool = True, **kwargs):
        if market_data.ticker not in self.subscription:
            return

        super().__call__(market_data=market_data, allow_out_session=allow_out_session, **kwargs)

    def __del__(self):
        self.unlink()

    def on_subscription(self, subscription: list[str] = None):
        if subscription:
            subscription = set(subscription) | set(self.subscription) if self.subscription else set()

            self.subscription.clear()
            self.subscription.extend(subscription)

        if isinstance(self, EMA):
            EMA.on_subscription(self=self)

        if isinstance(self, Synthetic):
            Synthetic.on_subscription(self=self)

        if isinstance(self, FixedIntervalSampler):
            FixedIntervalSampler.on_subscription(self=self)

        if isinstance(self, FixedVolumeIntervalSampler):
            FixedVolumeIntervalSampler.on_subscription(self=self)

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            AdaptiveVolumeIntervalSampler.on_subscription(self=self)

        return self.subscription

    def to_json(self, fmt='str', with_memory_core=False, **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)

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

        var_names = cls.__init__.__code__.co_varnames
        kwargs = {name: json_dict[name] for name in var_names if name in json_dict}

        self = cls(**kwargs)

        if 'subscription' in json_dict and 'subscription' not in kwargs:
            self.subscription = json_dict['subscription']

        self.update_from_json(json_dict=json_dict)

        return self

    def to_shm(self, name: str = None, manager: SharedMemoryCore = None) -> str:
        if name is None:
            name = f'{self.monitor_id}.json'

        if manager is None:
            manager = self.memory_core

        self.memory_core.to_shm()
        serialized = pickle.dumps(self)
        size = len(serialized)

        shm = manager.init_buffer(name=name, buffer_size=size, init_value=serialized)
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

        self.memory_core.from_shm()

        return self

    @property
    def use_shm(self) -> bool:
        if self.memory_core is None:
            return False
        elif self.memory_core.dummy:
            return False

        return True

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

                setattr(self, name, self.ema[name])

        if isinstance(self, Synthetic):
            self: Synthetic

            self.base_price.update(json_dict['base_price'])
            self.last_price.update(json_dict['last_price'])
            self.synthetic_base_price.value = json_dict['synthetic_base_price']

        if isinstance(self, FixedIntervalSampler):
            self: FixedIntervalSampler

            for name, sampler in json_dict['sample_storage'].items():
                mode = sampler['mode']
                new_sampler = self.register_sampler(name=name, mode=mode)
                new_sampler['index'].update(sampler['index'])

                for ticker, data in sampler['storage'].items():
                    if ticker in new_sampler:
                        new_sampler['storage'][ticker].extend(data)
                    else:
                        new_sampler['storage'][ticker] = self.memory_core.register(data, name=f'Sampler.{name}.{ticker}', dtype='Deque', maxlen=self.sample_size)

        if isinstance(self, FixedVolumeIntervalSampler):
            self: FixedVolumeIntervalSampler

            self._accumulated_volume.update(json_dict['accumulated_volume'])

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            self: AdaptiveVolumeIntervalSampler

            for name, value in json_dict['sample_storage'].items():
                sampler = self.sample_storage[name]
                sampler['index_vol'].update(value['index_vol'])

            self._volume_baseline['baseline'].update(json_dict['volume_baseline']['baseline'])
            self._volume_baseline['sampling_interval'].update(json_dict['volume_baseline']['sampling_interval'])
            self._volume_baseline['obs_vol_acc_start'].update(json_dict['volume_baseline']['obs_vol_acc_start'])
            self._volume_baseline['obs_index'].update(json_dict['volume_baseline']['obs_index'])
            for ticker, data in json_dict['volume_baseline']['obs_vol_acc'].items():
                if ticker in self._volume_baseline['obs_vol_acc']:
                    self._volume_baseline['obs_vol_acc'][ticker].extend(data)
                else:
                    self._volume_baseline['obs_vol_acc'][ticker] = self.memory_core.register(data, name=f'sampler.obs_vol_acc.{ticker}', dtype='Deque', maxlen=self.baseline_window)

        self.on_subscription()

        return self

    def _param_static(self) -> dict[str, ...]:
        param_static = super()._param_static()
        param_static['subscription'] = self.subscription
        return param_static

    def unlink(self):
        LOGGER.info(f'Unlinking monitor {self.name}, id={self.monitor_id}...')
        self.memory_core.unlink()

    @property
    def is_sync(self) -> bool:
        if self.memory_core.dummy:
            return True
        else:
            return self.memory_core.is_sync


class ConcurrentMonitorManager(_CMM):
    def __init__(self, subscription: list[str] = None, n_worker: int = None):
        super().__init__(n_worker=n_worker)
        self.subscription = [] if subscription is None else subscription

    def add_monitor(self, monitor: FactorMonitor):
        # init shared memory before adding to child process
        # this optional action will conserve io and resources
        monitor.on_subscription(self.subscription)
        # otherwise SyncTemplate.new_ver() will be called multiple times before all the required share memory is allocated.
        # on each time SyncTemplate.new_ver() called, to_shm() and from_shm() must be called to assure synchronization.
        # the corresponded codes are commented in _task_concurrent and worker function. Use with caution.

        super().add_monitor(monitor=monitor)

    def pop_monitor(self, monitor: FactorMonitor):
        monitor: FactorMonitor = super().add_monitor(monitor=monitor)
        monitor.unlink()

    def _task_concurrent(self, market_data: MarketData):
        main_tasks = []
        child_tasks = []

        # step 1: send market data to the shared memory
        self.tasks['md_dtype'].value = md_dtype = market_data.__class__.__name__
        self.tasks[md_dtype].update(market_data=market_data)

        # step 2: send monitor data core to shared memory
        for monitor_id, monitor in self.monitor.items():
            monitor: FactorMonitor
            if not monitor.enabled:
                continue
            elif monitor.serializable:
                # monitor.memory_core.to_shm()
                child_tasks.append(monitor_id)
            else:
                main_tasks.append(monitor_id)

        # step 3: release semaphore to enable the workers
        tasks = {worker_id: [] for worker_id in range(self.n_worker)}
        for task_id, monitor_id in enumerate(child_tasks):
            worker_id = task_id % self.n_worker
            tasks[worker_id].append(monitor_id)

        for worker_id, task_list in tasks.items():
            self.tasks['tasks'][worker_id].value = '\x03'.join(task_list)

        for _ in range(self.n_worker):
            self.semaphore_start.release()

        # step 4: execute tasks in main thread for those monitors not supporting multiprocessing features
        for monitor_id in main_tasks:
            self._work(monitor_id=monitor_id, market_data=market_data)

        # step 5: acquire semaphore to wait till the tasks all done
        for _ in range(self.n_worker):
            self.semaphore_done.acquire()

        if self.verbose and self.progress % 100000 == 0:
            self.progress += 1
            LOGGER.info(f'Progress {self.progress - 100000} to {self.progress} working time for the workers:\n{pd.Series([_.value for _ in self.tasks["time_cost"]]).to_string()}')
            for time_cost in self.tasks['time_cost']:
                time_cost.value = 0

        # step 6: update the monitor from shared memory
        # for monitor_id in child_tasks:
        #     monitor = self.monitor.get(monitor_id)
        #     if monitor is not None and monitor.enabled:
        #         monitor.memory_core.from_shm()

    def worker(self, worker_id: int):
        while True:
            self.semaphore_start.acquire()
            ts = time.time()
            # management job 0: terminate the worker on signal
            if not self.enabled.value:
                break

            # step 1.1: reconstruct market data
            md_dtype = self.tasks['md_dtype'].value
            task_list = self.tasks['tasks'][worker_id].value.split('\x03')
            market_data = self.tasks[md_dtype].to_market_data()

            # step 2: do the tasks
            for monitor_id in task_list:
                if not monitor_id:
                    continue

                if monitor_id not in self.monitor:
                    try:
                        self.monitor[monitor_id] = monitor = FactorMonitor.from_shm(monitor_id=monitor_id)
                        LOGGER.info(f'Worker {worker_id} loaded monitor {monitor.name} {monitor_id} from shared memory successfully.')
                    except FileNotFoundError as _:
                        LOGGER.error(f'Monitor {monitor_id} not found in shared memory.')
                        continue
                    except Exception as _:
                        LOGGER.error(f'Deserialize monitor {monitor_id} failed, traceback:\n{traceback.format_exc()}')
                        continue
                else:
                    monitor = self.monitor[monitor_id]

                # monitor.memory_core.from_shm()
                self._work(monitor_id=monitor_id, market_data=market_data)
                # monitor.memory_core.to_shm()

            self.tasks['time_cost'][worker_id].value += time.time() - ts
            self.semaphore_done.release()

    def get_values(self) -> dict[str, float]:
        values = collect_factor(self.monitor)
        return values


class EMA(_EMA, metaclass=abc.ABCMeta):
    def __init__(self, alpha: float = None, window: int = None, subscription: list[str] = None, memory_core: SyncMemoryCore = None):
        super().__init__(alpha=alpha, window=window)

        self.subscription: list[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self._ema_memory: dict[str, NamedVector] = getattr(self, '_ema_memory', {})
        self._ema_current: dict[str, NamedVector] = getattr(self, '_ema_current', {})
        self.ema: dict[str, NamedVector] = getattr(self, 'ema', {})

    def register_ema(self, name: str):
        if name in self.ema:
            LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
            return self.ema[name]

        self._ema_memory[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.memory', dtype='NamedVector')
        self._ema_current[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.current', dtype='NamedVector')
        self.ema[name]: NamedVector = self.memory_core.register(name=f'ema.{name}.value', dtype='NamedVector')

        return self.ema[name]

    def on_subscription(self, subscription: list[str] = None):
        if subscription:
            subscription = set(subscription) | set(self.subscription) if self.subscription else set()
            self.subscription.clear()
            self.subscription.extend(subscription)

        for name in self.ema:
            for ticker in self.subscription:
                if ticker in self._ema_memory[name]:
                    continue

                self._ema_memory[name][ticker] = np.nan
                self._ema_current[name][ticker] = 0.  # this can set any value, but 0. is preferred for better compatibility with accumulative_ema
                self.ema[name][ticker] = np.nan

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

        for name in json_dict['ema']:
            self.register_ema(name=name)

            self._ema_memory[name].update(json_dict['ema_memory'][name])
            self._ema_current[name].update(json_dict['ema_current'][name])
            self.ema[name].update(json_dict['ema'][name])

        return self

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
        super().__init__(weights=weights)

        self.subscription: list[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self.base_price: NamedVector = self.memory_core.register(name='base_price', dtype='NamedVector')
        self.last_price: NamedVector = self.memory_core.register(name='last_price', dtype='NamedVector')
        self.synthetic_base_price: FloatValue = self.memory_core.register(name='synthetic_base_price', dtype='FloatValue', size=8, value=1.)

    def on_subscription(self, subscription: list[str] = None):
        if subscription:
            subscription = set(subscription) | set(self.subscription) if self.subscription else set()
            self.subscription.clear()
            self.subscription.extend(subscription)

        for ticker in self.subscription:
            if ticker not in self.base_price:
                self.base_price[ticker] = np.nan

            if ticker not in self.last_price:
                self.last_price[ticker] = np.nan

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
        self.base_price.unlink()
        self.last_price.unlink()
        self.synthetic_base_price.value = 1

        self.base_price: NamedVector = self.memory_core.register(name='base_price', dtype='NamedVector')
        self.last_price: NamedVector = self.memory_core.register(name='last_price', dtype='NamedVector')

        self.on_subscription()
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
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size)

        self.subscription: list[str] = getattr(self, 'subscription', subscription)
        self.memory_core: SyncMemoryCore = getattr(self, 'memory_core', memory_core)

        if self.memory_core is None:
            self.memory_core = SyncMemoryCore(dummy=True)

        self.sample_storage = getattr(self, 'sample_storage', {})  # to avoid de-reference the dict using nested inheritance

    def register_sampler(self, name: str, mode: str | SamplerMode = 'update'):
        if name in self.sample_storage:
            LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
            return self.sample_storage[name]

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
        super().on_subscription(subscription=subscription)

        for ticker in self.subscription:
            if ticker in self._accumulated_volume:
                continue

            self._accumulated_volume[ticker] = 0.

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
            'baseline': self.memory_core.register(name='sampler.baseline', dtype='NamedVector'),
            'sampling_interval': self.memory_core.register(name='sampler.sampling_interval', dtype='NamedVector'),
            'obs_vol_acc_start': self.memory_core.register(name='sampler.obs_vol_acc_start', dtype='NamedVector'),
            'obs_index': self.memory_core.register(name='sampler.obs_index', dtype='NamedVector'),
            'obs_vol_acc': {},
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

        if not isinstance(sample_storage['index_vol'], NamedVector):
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
