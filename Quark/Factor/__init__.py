from __future__ import annotations

import abc
import json
import os
import pickle
import traceback
from collections import deque
from ctypes import c_wchar, c_bool
from functools import partial
from multiprocessing import shared_memory, RawArray, RawValue, Semaphore, Process
from typing import Iterable, Self

from AlgoEngine.Engine import MarketDataMonitor, MDS, MonitorManager
from PyQuantKit import MarketData, TickData, TradeData, TransactionData, OrderBook, OrderBookBuffer, BarDataBuffer, TickDataBuffer, TransactionDataBuffer

from .. import LOGGER
from ..Base import GlobalStatics
from ..Calibration.dummies import is_market_session

LOGGER = LOGGER.getChild('Factor')
TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE


class FactorMonitor(MarketDataMonitor, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: str = None):
        """
        the init function should never accept *arg or **kwargs as parameters.
        since the from_json function depends on it.
        """
        var_names = self.__class__.__init__.__code__.co_varnames

        if 'kwargs' in var_names or 'args' in var_names:
            LOGGER.warning(f'*args and **kwargs are should not in {self.__class__.__name__} initialization. All parameters must be explicit.')

        assert name.startswith('Monitor')
        super().__init__(name=name, monitor_id=monitor_id)

    def __call__(self, market_data: MarketData, allow_out_session: bool = True, **kwargs):
        # filter the out session data
        if not (is_market_session(market_data.timestamp) or allow_out_session):
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

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            name=self.name,
            monitor_id=self.monitor_id
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

    def to_shm(self, name: str = None) -> str:
        if name is None:
            name = f'{self.monitor_id}.json'

        return super().to_shm(name=name)

    def from_shm(self, name: str = None) -> None:
        if name is None:
            name = f'{self.monitor_id}.json'

        shm = shared_memory.SharedMemory(name=name)
        json_dict = pickle.loads(bytes(shm.buf))
        shm.close()
        self.update_from_json(json_dict=json_dict)

    def update_from_json(self, json_dict: dict) -> Self:
        """
        a utility function for .from_json()
        """
        if isinstance(self, FixedIntervalSampler):
            self.sample_storage.clear()
            self.sample_storage.update(json_dict['sample_storage'])

        if isinstance(self, FixedVolumeIntervalSampler):
            self._accumulated_volume.clear()
            self._accumulated_volume.update(json_dict['accumulated_volume'])

        if isinstance(self, AdaptiveVolumeIntervalSampler):
            self._volume_baseline.clear()
            self._volume_baseline.update(json_dict['volume_baseline'])

        if isinstance(self, Synthetic):
            self.weights.clear()
            self.weights.index_name = json_dict['index_name']
            self.weights.update(json_dict['weights'])
            self.base_price.clear()
            self.base_price.update(json_dict['base_price'])
            self.last_price.clear()
            self.last_price.update(json_dict['last_price'])
            self.synthetic_base_price = json_dict['synthetic_base_price']

        if isinstance(self, EMA):
            self._last_discount_ts.clear()
            self._last_discount_ts.update(json_dict['last_discount_ts'])
            self._history.clear()
            self._history.update(json_dict['history'])
            self._current.clear()
            self._current.update(json_dict['current'])
            self._window.clear()
            self._window.update({entry_name: {ticker: deque(data, maxlen=self.window) for ticker, data in entry} for entry_name, entry in json_dict['window_deque'].items()})
            self.ema.clear()
            self.ema.update(json_dict['ema'])

            for name in self.ema:
                setattr(self, name, self.ema[name])

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

        if isinstance(self, EMA):
            param_static.update(
                discount_interval=self.discount_interval
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


class ConcurrentMonitorManager(MonitorManager):
    def __init__(self, n_worker: int = None):
        super().__init__()

        self.n_worker = os.cpu_count() - 1 if n_worker is None else n_worker
        self.workers = []
        self.tasks = {
            'md_dtype': RawArray(c_wchar, 16),
            'tasks': [RawArray(c_wchar, 8 * 1024) for _ in range(self.n_worker)],
            'OrderBook': OrderBookBuffer(),
            'BarData': BarDataBuffer(),
            'TickData': TickDataBuffer(),
            'TransactionData': TransactionDataBuffer(),
            'TradeData': TransactionDataBuffer()
        }
        self.enabled = RawValue(c_bool, False)
        self.semaphore_start = Semaphore(value=0)
        self.semaphore_done = Semaphore(value=0)

        if self.n_worker <= 1:
            LOGGER.info('Monitor manager is initialized in single process mode!')
        else:
            LOGGER.info(f'Monitor manager is initialized with {self.n_worker} processes mode!')
            self.init_process()

    def add_monitor(self, monitor: FactorMonitor):
        super().add_monitor(monitor=monitor)

        if self.workers:
            try:
                serialized = pickle.dumps(monitor)
                size = len(serialized)
                shm = shared_memory.SharedMemory(name=f'{monitor.monitor_id}.pickle', create=True, size=size)
                shm.buf[:] = serialized
                shm.close()
            except Exception as _:
                LOGGER.info(f'Monitor {monitor.name} serialization failed, multiprocessing not available.')

    def pop_monitor(self, monitor_id: str):
        super().pop_monitor(monitor_id=monitor_id)

        if self.workers:
            try:
                shm = shared_memory.SharedMemory(name=monitor_id)
                shm.close()
                shm.unlink()
            except FileNotFoundError as _:
                LOGGER.debug(f'No monitor with id {monitor_id} in shared memory.')

    def __call__(self, market_data: MarketData):
        # multi processing mode
        if self.workers:
            self._task_concurrent(market_data=market_data)
        # single processing mode
        else:
            super().__call__(market_data=market_data)

    def _task_concurrent(self, market_data: MarketData):
        main_tasks = []
        child_tasks = []

        # step 1: send market data to the shared memory
        self.tasks['md_dtype'].value = md_dtype = market_data.__class__.__name__
        self.tasks[md_dtype].update(market_data=market_data)

        # step 2: send monitor data core to shared memory
        for monitor_id, monitor in self.monitor.items():
            try:
                monitor.to_shm()
                child_tasks.append(monitor_id)
            except Exception as _:
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

        # step 6: update the monitor from shared memory
        for monitor_id in child_tasks:
            monitor = self.monitor.get(monitor_id)
            if monitor is not None and monitor.enabled:
                monitor.from_shm()

    def worker(self, worker_id: int):
        while True:
            self.semaphore_start.acquire()

            # management job 0: terminate the worker on signal
            if not self.enabled.value:
                break

            # step 1.1: reconstruct market data
            md_dtype = self.tasks['md_dtype'].value
            task_list = self.tasks['tasks'][worker_id].value.split('\x03')
            market_data = self.tasks[md_dtype].to_market_data()

            # step 2: do the tasks
            for monitor_id in task_list:

                if monitor_id not in self.monitor:
                    try:
                        LOGGER.debug(f'Worker {worker_id} can not find monitor with id {monitor_id}, looking up in shared memory...')
                        shm = shared_memory.SharedMemory(name=f'{monitor_id}.pickle')
                        monitor = pickle.loads(bytes(shm.buf))
                        self.monitor[monitor_id] = monitor
                        shm.close()
                    except Exception as _:
                        LOGGER.error(f'Deserialize monitor {monitor_id} failed, traceback:\n{traceback.format_exc()}')
                        continue

                self.monitor[monitor_id].from_shm()
                self._work(monitor_id=monitor_id, market_data=market_data)
                self.monitor[monitor_id].to_shm()

            self.semaphore_done.release()

    def init_process(self):
        self.enabled.value = True

        for i in range(self.n_worker):
            p = Process(target=self.worker, name=f'{self.__class__.__name__}.worker.{i}', kwargs={'worker_id': i})
            p.start()
            self.workers.append(p)

    def clear(self):
        self.enabled.value = False

        for _ in range(self.n_worker):
            self.semaphore_start.release()

        for p in self.workers:
            p.join()

        self.workers.clear()
        self.enabled.value = False

        while self.semaphore_start.acquire(False):
            continue

        while self.semaphore_done.acquire(False):
            continue

        # clear shm
        for monitor_id in self.monitor:
            try:
                shm = shared_memory.SharedMemory(name=f'{monitor_id}.pickle')
                shm.close()
                shm.unlink()
            except FileNotFoundError as _:
                continue

        super().clear()


def add_monitor(monitor: FactorMonitor, **kwargs) -> dict[str, FactorMonitor]:
    monitors = kwargs.get('monitors', {})
    factors = kwargs.get('factors', None)
    register = kwargs.get('register', True)

    if factors is None:
        is_pass_check = True
    elif isinstance(factors, str) and factors == monitor.name:
        is_pass_check = True
    elif isinstance(factors, Iterable) and monitor.name in factors:
        is_pass_check = True
    else:
        return monitors

    if is_pass_check:
        monitors[monitor.name] = monitor

        if register:
            MDS.add_monitor(monitor)

    return monitors


from .utils import *

ALPHA_05 = 0.9885  # alpha = 0.5 for each minute
ALPHA_02 = 0.9735  # alpha = 0.2 for each minute
ALPHA_01 = 0.9624  # alpha = 0.1 for each minute
ALPHA_001 = 0.9261  # alpha = 0.01 for each minute
ALPHA_0001 = 0.8913  # alpha = 0.001 for each minute
INDEX_WEIGHTS = IndexWeight(index_name='DummyIndex')
MONITOR_MANAGER = ConcurrentMonitorManager()

from .TradeFlow import *
from .Correlation import *
from .Distribution import *
from .Misc import *
from .LowPass import *
from .Decoder import *


def register_monitor(**kwargs) -> dict[str, FactorMonitor]:
    monitors = kwargs.get('monitors', {})
    index_name = kwargs.get('index_name', 'SyntheticIndex')
    index_weights = IndexWeight(index_name=index_name, **kwargs.get('index_weights', INDEX_WEIGHTS))
    factors = kwargs.get('factors', None)

    index_weights.normalize()
    check_and_add = partial(add_monitor, factors=factors, monitors=monitors)
    LOGGER.info(f'Register monitors for index {index_name} and its {len(index_weights.components)} components!')

    # trade flow monitor
    check_and_add(TradeFlowMonitor())

    # trade flow ema monitor
    check_and_add(TradeFlowEMAMonitor(discount_interval=1, alpha=ALPHA_05, weights=index_weights))

    # price coherence monitor
    check_and_add(CoherenceMonitor(sampling_interval=1, sample_size=60, weights=index_weights))

    # price coherence ema monitor
    check_and_add(CoherenceEMAMonitor(sampling_interval=1, sample_size=60, weights=index_weights, discount_interval=1, alpha=ALPHA_0001))

    # trade coherence monitor
    check_and_add(TradeCoherenceMonitor(sampling_interval=1, sample_size=60, weights=index_weights))

    # synthetic index monitor
    check_and_add(SyntheticIndexMonitor(index_name=index_name, weights=index_weights))

    # aggressiveness monitor
    check_and_add(AggressivenessMonitor())

    # aggressiveness ema monitor
    check_and_add(AggressivenessEMAMonitor(discount_interval=1, alpha=ALPHA_0001, weights=index_weights))

    # price coherence monitor
    check_and_add(EntropyMonitor(sampling_interval=1, sample_size=60, weights=index_weights))

    # price coherence monitor
    check_and_add(EntropyEMAMonitor(sampling_interval=1, sample_size=60, weights=index_weights, discount_interval=1, alpha=ALPHA_0001))

    # price coherence monitor
    check_and_add(VolatilityMonitor(weights=index_weights))

    # price movement online decoder
    check_and_add(DecoderMonitor(retrospective=False))

    # price movement online decoder
    check_and_add(IndexDecoderMonitor(up_threshold=0.005, down_threshold=0.005, confirmation_level=0.002, retrospective=True, weights=index_weights))

    return monitors


def collect_factor(monitors: dict[str, FactorMonitor] | list[FactorMonitor] | FactorMonitor) -> dict[str, float]:
    factors = {}

    if isinstance(monitors, dict):
        monitors = list(monitors.values())
    elif isinstance(monitors, FactorMonitor):
        monitors = [monitors]

    for monitor in monitors:
        if monitor.is_ready and monitor.enabled:
            factor_value = monitor.value
            name = monitor.name.removeprefix('Monitor.')

            if isinstance(factor_value, (int, float)):
                factors[name] = factor_value
            elif isinstance(factor_value, dict):
                # FactorPoolDummyMonitor having hard coded name
                if monitor.name == 'Monitor.FactorPool.Dummy':
                    factors.update(factor_value)
                # synthetic index monitor should have duplicated logs
                elif isinstance(monitor, SyntheticIndexMonitor):
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
                    factors.update({f'{monitor.index_name}.{key}': value for key, value in factor_value.items()})
                else:
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
            else:
                raise NotImplementedError(f'Invalid return type, expect float | dict[str, float], got {type(factor_value)}.')

    return factors


__all__ = [
    'FactorMonitor', 'LOGGER', 'DEBUG_MODE', 'IndexWeight', 'MONITOR_MANAGER', 'Synthetic', 'EMA', 'collect_factor',
    # from .Correlation module
    'CoherenceMonitor', 'CoherenceAdaptiveMonitor', 'CoherenceEMAMonitor', 'TradeCoherenceMonitor', 'EntropyMonitor', 'EntropyAdaptiveMonitor', 'EntropyEMAMonitor',
    # from Decoder module
    'DecoderMonitor', 'IndexDecoderMonitor', 'VolatilityMonitor',
    # from Distribution module
    'SkewnessMonitor', 'SkewnessIndexMonitor', 'SkewnessAdaptiveMonitor', 'SkewnessIndexAdaptiveMonitor', 'GiniMonitor', 'GiniIndexMonitor', 'GiniAdaptiveMonitor', 'GiniIndexAdaptiveMonitor',
    # from LowPass module
    'TradeClusteringMonitor', 'TradeClusteringAdaptiveMonitor', 'TradeClusteringIndexAdaptiveMonitor', 'DivergenceMonitor', 'DivergenceAdaptiveMonitor', 'DivergenceIndexAdaptiveMonitor',
    # from Misc module
    'SyntheticIndexMonitor',
    # from TradeFlow module
    'AggressivenessMonitor', 'AggressivenessEMAMonitor', 'TradeFlowMonitor', 'TradeFlowAdaptiveMonitor', 'TradeFlowAdaptiveIndexMonitor'
]
