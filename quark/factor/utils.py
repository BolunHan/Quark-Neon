from __future__ import annotations

import abc
import enum
import json
import pickle
import time
import traceback
from collections import deque
from ctypes import c_wchar, c_bool, c_double
from multiprocessing import RawArray, RawValue, Semaphore, Process, shared_memory, Lock
from typing import Self

import numpy as np
import pandas as pd
from algo_engine.base import MarketData, TickData, TradeData, TransactionData, OrderBook, BarData, OrderBookBuffer, BarDataBuffer, TickDataBuffer, TransactionDataBuffer
from algo_engine.engine import MarketDataMonitor, MonitorManager

from . import collect_factor
from .memory_core import SharedMemoryCore, SyncMemoryCore, NamedVector
from .. import LOGGER
from ..base import GlobalStatics
import hashlib

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
            manager = SharedMemoryCore()

        serialized = pickle.dumps(self)
        size = len(serialized)

        shm = manager.init_buffer(name=name, buffer_size=size)
        shm.buf[:] = serialized
        shm.close()
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
        self: Self = pickle.loads(bytes(shm.buf))

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

            for name, sampler in json_dict['sample_storage'].items():
                mode = sampler['mode']
                new_sampler = self.register_sampler(name=name, mode=mode)
                new_sampler['index'].update(sampler['index'])

                for ticker, data in sampler['storage'].items():
                    if ticker in new_sampler:
                        new_sampler['storage'][ticker].extend(data)
                    else:
                        new_sampler['storage'][ticker] = deque(data, maxlen=self.sample_size)

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
        memory_core: SyncMemoryCore = getattr(self, 'memory_core', None)

        if memory_core is None:
            return False
        elif memory_core.dummy:
            return False

        return True

    @property
    def meta(self) -> dict[str, str | float | int | bool]:
        meta_info = self._update_meta_info(meta_info=self.__additional_meta_info)
        return {k: meta_info[k] for k in sorted(meta_info)}

    def digest(self, encoding: str = 'utf-8') -> str:
        hashed_str = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode(encoding=encoding)).hexdigest()
        return hashed_str


class ConcurrentMonitorManager(MonitorManager):
    def __init__(self, n_worker: int, verbose: bool = True):
        super().__init__()

        self.n_worker = n_worker
        self.manager = SyncMemoryCore(prefix='MonitorManager')
        self.verbose = verbose

        self.workers = []
        self.child_monitor = {}
        self.main_monitor = {}
        self.tasks = {
            'md_dtype': RawArray(c_wchar, 16),
            'monitor_value': self.manager.register(name='monitor_value', dtype='NamedVector'),
            'tasks': [RawArray(c_wchar, 8 * 1024) for _ in range(self.n_worker)],
            'time_cost': [RawValue(c_double, 0) for _ in range(self.n_worker)],
            'OrderBook': OrderBookBuffer(),
            'BarData': BarDataBuffer(),
            'TickData': TickDataBuffer(),
            'TransactionData': TransactionDataBuffer(),
            'TradeData': TransactionDataBuffer()
        }
        self.enabled = RawValue(c_bool, False)
        self.request_value = RawValue(c_bool, False)
        self.lock = Lock()
        self.semaphore_start = Semaphore(value=0)
        self.semaphore_done = Semaphore(value=0)
        self.progress = 0

        if self.n_worker <= 1:
            LOGGER.info(f'{self.__class__.__name__} is set as single process mode!')
        else:
            LOGGER.info(f'{self.__class__.__name__} is set as {self.n_worker} processes mode!')

    def add_monitor(self, monitor: FactorMonitor):
        super().add_monitor(monitor=monitor)

        if self.n_worker <= 1:
            LOGGER.info(f'Assign monitor {monitor.name} to main worker.')
            self.main_monitor[monitor.monitor_id] = monitor
        elif monitor.serializable:
            monitor_id = monitor.monitor_id
            worker_id = len(self.child_monitor) % self.n_worker
            self.tasks['tasks'][worker_id].value += f'\x03{monitor_id}'
            self.child_monitor[monitor_id] = monitor
            monitor.to_shm(manager=self.manager)
            LOGGER.info(f'Assign monitor {monitor.name} to worker {worker_id}, serialized in shm "{monitor_id}.json".')
        else:
            LOGGER.debug(f'Monitor {monitor.name} is marked as not serializable. Assigned to the main process.')
            self.main_monitor[monitor.monitor_id] = monitor

    def pop_monitor(self, monitor_id: str):
        if monitor_id in self.main_monitor:
            self.main_monitor.pop(monitor_id, None)
        elif monitor_id in self.child_monitor:
            self.child_monitor.pop(monitor_id, None)

            try:
                shm = shared_memory.SharedMemory(name=f'{monitor_id}.json')
                shm.close()
                shm.unlink()
            except FileNotFoundError as _:
                pass
        else:
            LOGGER.warning(f'Monitor id {monitor_id} not found!')

        return self.monitor.pop(monitor_id)

    def __call__(self, market_data: MarketData):
        if self.workers:
            self._task_concurrent(market_data=market_data)
        else:
            super().__call__(market_data=market_data)

    def _task_concurrent(self, market_data: MarketData):
        # step 1: send market data to the shared memory
        self.tasks['md_dtype'].value = md_dtype = market_data.__class__.__name__
        self.tasks[md_dtype].update(market_data=market_data)

        # step 2: release the semaphore
        for _ in self.workers:
            self.semaphore_start.release()

        # step 3: execute tasks in main thread for those monitors not supporting multiprocessing features
        for monitor_id in self.main_monitor:
            self._work(monitor_id=monitor_id, market_data=market_data)

        # step 4: acquire semaphore to wait till the tasks all done
        for _ in self.workers:
            self.semaphore_done.acquire()

        # step 5: log some performance metrics
        if self.verbose and self.progress and self.progress % 100000 == 0:
            self.progress += 1
            LOGGER.info(f'Progress {self.progress - 100000} to {self.progress} working time for the workers:\n{pd.Series([_.value for _ in self.tasks["time_cost"]]).to_string()}')
            for time_cost in self.tasks['time_cost']:
                time_cost.value = 0

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

            # when requesting monitor value, the market data will not be updated.
            if self.request_value.value:

                child_monitor = []
                for monitor_id in task_list:
                    if not monitor_id:
                        continue
                    elif monitor_id not in self.monitor:
                        continue

                    child_monitor.append(self.monitor[monitor_id])

                monitor_value_vector: NamedVector = self.tasks['monitor_value']
                updated_value = collect_factor(child_monitor)

                if updated_value:
                    with self.lock:
                        monitor_value_vector.from_shm()
                        monitor_value_vector.update(updated_value)
                        monitor_value_vector.to_shm()
            else:
                market_data = self.tasks[md_dtype].to_market_data()
                # step 2: do the tasks
                for monitor_id in task_list:
                    if not monitor_id:
                        continue

                    if monitor_id not in self.monitor:
                        try:
                            self.monitor[monitor_id] = monitor = FactorMonitor.from_shm(monitor_id=monitor_id, manager=self.manager)
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

    def start(self):
        if self.enabled.value:
            LOGGER.warning(f'{self.__class__.__name__} already started!')
            return

        self.enabled.value = True
        self.progress = 0

        if self.n_worker <= 1:
            LOGGER.info(f'{self.__class__.__name__} in single process mode, and will not spawn any child process.')
            return

        main_monitor = self.main_monitor
        child_monitor = self.child_monitor
        delattr(self, 'main_monitor')
        delattr(self, 'child_monitor')

        self.monitor.clear()

        for worker_id in range(self.n_worker):
            task_list = self.tasks['tasks'][worker_id].value.split('\x03')

            for monitor_id in task_list:
                if not monitor_id:
                    continue
                self.monitor[monitor_id] = child_monitor[monitor_id]

            if not self.monitor:
                continue

            p = Process(target=self.worker, name=f'{self.__class__.__name__}.worker.{worker_id}', kwargs={'worker_id': worker_id})
            p.start()
            self.workers.append(p)

            self.monitor.clear()

        self.main_monitor = main_monitor
        self.child_monitor = child_monitor

        self.monitor.update(main_monitor)
        self.monitor.update(child_monitor)

        LOGGER.info(f'{self.__class__.__name__} started {"all " if len(self.workers) == self.n_worker else ""}{len(self.workers)} workers.')

    def stop(self):
        self.enabled.value = False

        for _ in self.workers:
            self.semaphore_start.release()

        for i, p in enumerate(self.workers):
            p.join()
            p.close()

        self.workers.clear()

        while self.semaphore_start.acquire(False):
            continue

        while self.semaphore_done.acquire(False):
            continue

        LOGGER.info(f'{self.__class__.__name__} all worker stopped and cleared.')

    def clear(self):
        # clear shm
        for monitor_id in list(self.monitor):
            self.pop_monitor(monitor_id)

        super().clear()

        # this is a monkey patch for windows shm management.
        # Python does not release shared memory on unlink() and will cause register failed.
        # this is the reason way multiprocessing is disabled on windows platform
        if self.n_worker <= 1:
            pass
        else:
            self.tasks['monitor_value'].unlink()
            self.tasks['monitor_value'] = self.manager.register(name='monitor_value', dtype='NamedVector')

        for _ in self.tasks['time_cost']:
            _.value = 0

        for _ in self.tasks['tasks']:
            _.value = ''

        # self.manager.unlink()

    def get_values(self) -> dict[str, float]:
        self.request_value.value = True

        for _ in self.workers:
            self.semaphore_start.release()

        monitor_value = collect_factor(self.main_monitor)

        for _ in self.workers:
            self.semaphore_done.acquire()

        self.request_value.value = False

        monitor_value_vector: NamedVector = self.tasks['monitor_value']
        monitor_value_vector.from_shm()
        monitor_value.update(monitor_value_vector)

        return monitor_value

    @property
    def values(self) -> dict[str, float]:
        if self.workers:
            return self.get_values()
        else:
            return collect_factor(self.monitor)


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

    def __init__(self, sampling_interval: float = 1., sample_size: int = 60):
        """
        Initialize the FixedIntervalSampler.

        Parameters:
        - sampling_interval (float): Time interval between consecutive samples (in seconds). Default is 1.
        - sample_size (int): Number of samples to be stored. Default is 60.
        """
        self.sampling_interval = sampling_interval
        self.sample_size = sample_size

        self.sample_storage = getattr(self, 'sample_storage', {})  # to avoid de-reference the dict using nested inheritance

        # Warning for sampling_interval
        if sampling_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive sampling_interval')

        # Warning for sample_interval by Shannon's Theorem
        if sample_size <= 2:
            LOGGER.warning(f"{self.__class__.__name__} should have a larger sample_size, by Shannon's Theorem, sample_size should be greater than 2")

    def register_sampler(self, name: str, mode: str | SamplerMode = 'update') -> dict:
        if name in self.sample_storage:
            LOGGER.warning(f'name {name} already registered in {self.__class__.__name__}!')
            return self.sample_storage[name]

        if isinstance(mode, SamplerMode):
            mode = mode.value

        if mode not in ['update', 'max', 'min', 'accumulate']:
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

    def log_obs(self, ticker: str, timestamp: float, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
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
            elif auto_register:
                obs_storage = storage[ticker] = deque(maxlen=self.sample_size)
            else:
                LOGGER.warning(f'Ticker {ticker} not registered in sampler {obs_name}, perhaps the subscription has changed?')
                continue

            last_idx = indices.get(ticker, 0)

            if idx > last_idx:
                obs_storage.append(obs_value)
                indices[ticker] = idx
                self.on_entry_added(ticker=ticker, name=obs_name, value=obs_value)
            else:
                if mode == 'update':
                    last_obs = obs_storage[-1] = obs_value
                elif mode == 'max':
                    last_obs = obs_storage[-1] = max(obs_value, obs_storage[-1])
                elif mode == 'min':
                    last_obs = obs_storage[-1] = min(obs_value, obs_storage[-1])
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

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            sampling_interval=self.sampling_interval,
            sample_size=self.sample_size,
            sample_storage={name: dict(storage={ticker: list(dq) for ticker, dq in value['storage'].items()},
                                       index=value['index'],
                                       mode=value['mode'])
                            for name, value in self.sample_storage.items()}
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
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        for name, sampler in json_dict['sample_storage'].items():
            mode = sampler['mode']
            new_sampler = self.register_sampler(name=name, mode=mode)
            new_sampler['index'].update(sampler['index'])

            for ticker, data in sampler['storage'].items():
                if ticker in new_sampler:
                    new_sampler['storage'][ticker].extend(data)
                else:
                    new_sampler['storage'][ticker] = deque(data, maxlen=self.sample_size)

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
        if market_data is not None and (not GlobalStatics.PROFILE.is_market_session(market_data.timestamp)):
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

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
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

        super().log_obs(ticker=ticker, timestamp=volume_accumulated, observation=observation, auto_register=auto_register, **kwargs)

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

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size']
        )

        for name, sampler in json_dict['sample_storage'].items():
            mode = sampler['mode']
            new_sampler = self.register_sampler(name=name, mode=mode)
            new_sampler['index'].update(sampler['index'])

            for ticker, data in sampler['storage'].items():
                if ticker in new_sampler:
                    new_sampler['storage'][ticker].extend(data)
                else:
                    new_sampler['storage'][ticker] = deque(data, maxlen=self.sample_size)

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
            'obs_index': {},  # type: dict[str, dict[float,float]]
            'obs_vol_acc': {},  # type: dict[str, deque]
        }

    def register_sampler(self, name: str, mode='update'):
        sample_storage = super().register_sampler(name=name, mode=mode)

        if 'index_vol' not in sample_storage:
            sample_storage['index_vol'] = {}

        return sample_storage

    def _update_volume_baseline(self, ticker: str, timestamp: float, volume_accumulated: float = None, min_obs: int = None, auto_register: bool = True) -> float | None:
        """
        Updates and calculates the baseline volume for a given ticker.

        Parameters:
        - ticker (str): Ticker symbol for volume baseline calculation.
        - timestamp (float): Timestamp of the observation.
        - volume_accumulated (float): Accumulated volume at the provided timestamp.

        Returns:
        - float | None: Calculated baseline volume or None if the baseline is not ready.

        """
        min_obs = min(self.sample_size, int(self.baseline_window // 2)) if min_obs is None else min_obs

        volume_baseline_dict: dict[str, float] = self._volume_baseline['baseline']
        obs_vol_acc_start_dict: dict[str, float] = self._volume_baseline['obs_vol_acc_start']
        obs_index_dict: dict[str, float] = self._volume_baseline['obs_index']

        volume_baseline = volume_baseline_dict.get(ticker, np.nan)
        volume_accumulated = self._accumulated_volume.get(ticker, 0.) if volume_accumulated is None else volume_accumulated

        if np.isfinite(volume_baseline):
            return volume_baseline

        if ticker in (_ := self._volume_baseline['obs_vol_acc']):
            obs_vol_acc: deque = _[ticker]
        elif auto_register:
            obs_vol_acc = _[ticker] = deque(maxlen=self.baseline_window)
        else:
            LOGGER.warning(f'Ticker {ticker} not registered in {self.__class__.__name__}, perhaps the subscription has changed?')
            return None

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

    def log_obs(self, ticker: str, timestamp: float, volume_accumulated: float = None, observation: dict[str, ...] = None, auto_register: bool = True, **kwargs):
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
        volume_sampling_interval = self._update_volume_baseline(ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated, auto_register=auto_register)

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
            elif auto_register:
                obs_storage = storage[ticker] = deque(maxlen=self.sample_size)
            else:
                LOGGER.warning(f'Ticker {ticker} not registered in sampler {obs_name}, perhaps the subscription has changed?')
                continue

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
                elif mode == 'max':
                    last_obs = obs_storage[-1] = max(obs_value, obs_storage[-1])
                elif mode == 'min':
                    last_obs = obs_storage[-1] = min(obs_value, obs_storage[-1])
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
                baseline=self._volume_baseline['baseline'],
                sampling_interval=self._volume_baseline['sampling_interval'],
                obs_vol_acc_start=self._volume_baseline['obs_vol_acc_start'],
                obs_index=self._volume_baseline['obs_index'],
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

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval']
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
                    new_sampler['storage'][ticker] = deque(data, maxlen=self.sample_size)

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

    @property
    def baseline_ready(self) -> bool:
        subscription = set(self._volume_baseline['obs_vol_acc'])

        for ticker in subscription:
            sampling_interval = self._volume_baseline['sampling_interval'].get(ticker, np.nan)

            if not np.isfinite(sampling_interval):
                return False

        return True

    @property
    def baseline_stable(self) -> bool:
        subscription = set(self._volume_baseline['obs_vol_acc'])

        for ticker in subscription:
            sampling_interval = self._volume_baseline['baseline'].get(ticker, np.nan)

            if not np.isfinite(sampling_interval):
                return False

        return True


__all__ = ['FactorMonitor', 'ConcurrentMonitorManager',
           'EMA', 'ALPHA_05', 'ALPHA_02', 'ALPHA_01', 'ALPHA_001', 'ALPHA_0001',
           'Synthetic', 'IndexWeight',
           'SamplerMode', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler']
