from __future__ import annotations

import abc
import hashlib
import os
import pickle
from collections.abc import Iterable
from ctypes import c_char, c_bool, c_double, c_int, memset, Structure, c_ulonglong
from multiprocessing import RawArray, RawValue, Semaphore, Process, Lock, Condition, Barrier

from algo_engine.base import MarketData, MarketDataRingBuffer
from algo_engine.engine import MonitorManager

from . import collect_factor
from .utils import FactorMonitor
from .. import LOGGER

__all__ = ['MarketDataBuffer', 'ConcurrentMonitorManager', 'AsyncMonitorManager', 'SyncMonitorManager']


class MarketDataBuffer(object):
    def __init__(self, n_workers: int, size: int, block: bool = False):
        self._buffer: MarketDataRingBuffer = MarketDataRingBuffer(size=size, block=block)
        self._n_workers = n_workers

        self._processed_flags = RawArray(c_bool, self._n_workers * self._buffer.size)
        self._head = RawArray(c_int, self._n_workers)

        self.condition_get: Condition = self._buffer.condition_get
        self.condition_put: Condition = self._buffer.condition_put

    def _get(self, worker_id: int) -> MarketData:
        # get the index of the task
        index = self._head[worker_id]
        # get the market buffer
        md_buffer = self._buffer.at(index=index)
        # convert to the market data
        # market_data = md_buffer.to_market_data()
        market_data = md_buffer.contents
        # mark md is processed
        self._processed_flags[index * self._n_workers + worker_id] = True

        return market_data

    def get(self, worker_id: int) -> MarketData | None:
        while self.is_worker_empty(worker_id=worker_id):
            if not self._buffer.block:
                return None

            with self.condition_get:
                self.condition_get.wait()

        # same as the _get method
        index = self._head[worker_id]
        index_circled = index % self._buffer.size
        md_buffer = self._buffer.at(index=index)
        market_data = md_buffer.contents
        self._processed_flags[index_circled * self._n_workers + worker_id] = True

        if all(self._processed_flags[index_circled * self._n_workers: (index_circled + 1) * self._n_workers]):
            if self._buffer.block and self._buffer.is_full():
                self._head[worker_id] = self._buffer.head = index + 1
                self.condition_put.notify_all()
            else:
                self._head[worker_id] = self._buffer.head = index + 1
        else:
            self._head[worker_id] = index + 1

        return market_data

    def put(self, market_data: MarketData):
        while self._buffer.is_full():
            if not self._buffer.block:
                continue

            with self.condition_put:
                self.condition_put.wait()

        # update buffer
        self._buffer.put(market_data=market_data)

        if self._buffer.block:
            to_notify = False
            for worker_id in range(self._n_workers):
                if self.is_worker_empty(worker_id=worker_id):
                    to_notify = True
                    break

            if to_notify:
                with self.condition_get:
                    self.condition_get.notify_all()

    def is_worker_empty(self, worker_id: int) -> bool:
        return self._head[worker_id] == self._buffer.tail

    def is_empty(self):
        for worker_id in range(self._n_workers):
            if not self.is_worker_empty(worker_id):
                return False
        return True

    @property
    def block(self) -> bool:
        return self._buffer.block

    @property
    def head(self):
        return self._head[:]

    @property
    def tail(self):
        return self._buffer.tail


class FactorValueCollector(object):
    def __init__(self, entry_size: int = 1024, buffer_size: int = 1024, **kwargs):
        self._entry_size = entry_size
        self._buffer_size = buffer_size
        self._encoding = kwargs.get('encoding', 'utf-8')
        self._keys = kwargs['key_buffer'] if 'key_buffer' in kwargs else RawArray(c_char, self._entry_size * self._buffer_size)
        self._values = kwargs['value_buffer'] if 'value_buffer' in kwargs else RawArray(c_double, self._entry_size)
        self._length = kwargs['length_buffer'] if 'length_buffer' in kwargs else RawValue(c_ulonglong)
        self.lock = kwargs.get('lock')

    def __len__(self):
        return self._length.value

    def __getitem__(self, key: str) -> float:
        bytes_key = key.encode(self._encoding)
        key_list = list(self.keys())

        try:
            idx = key_list.index(key)
            return self._values[idx]
        except ValueError as _:
            raise KeyError(f'Key {key} not found!')

    def __iter__(self) -> Iterable[tuple[str, float]]:
        for i in range(self.__len__()):
            yield self._get_key(idx=i), self._values[i]

    def update(self, data: dict[str, float] = None, **kwargs):
        if self.lock is not None:
            self.lock.acquire()

        if data:
            patch = data | kwargs
        else:
            patch = kwargs

        key_list = list(self.keys())

        for _key, _value in patch.items():
            try:
                idx = key_list.index(_key)
                self._values[idx] = _value
            except ValueError as _:
                idx = self._length.value
                self._set_key(idx=idx, value=_key)
                self._values[idx] = _value
                self._length.value += 1

        if self.lock is not None:
            self.lock.release()

    def clear(self):
        self._length.value = 0

    def _get_key(self, idx: int) -> str:
        return self._keys[idx * self._buffer_size:(idx + 1) * self._buffer_size].rstrip(b'\x00').decode(self._encoding)

    def _set_key(self, idx: int, value: str):
        bytes_key = value.encode(self._encoding)
        self._keys[idx * self._buffer_size:(idx + 1) * self._buffer_size] = bytes_key.ljust(self._buffer_size, b'\x00')

    def keys(self) -> Iterable[str]:
        for i in range(self._length.value):
            yield self._get_key(idx=i)

    def values(self) -> Iterable[float]:
        for i in range(self._length.value):
            yield self._values[i]

    def items(self) -> Iterable[tuple[str, float]]:
        return self.__iter__()


class ConcurrentMonitorManager(MonitorManager):
    class Signature(Structure):
        _fields_ = [
            ('md5', c_char * 16)
        ]

    class Telemetries(Structure):
        _fields_ = [
            ('working_time', c_double),
            ('blocked_time', c_double),
            ('calculation_time', c_double),
            ('n_calls', c_int),
            ('n_loops', c_int),
        ]

    def __init__(self, **kwargs):
        super().__init__()

        self.enable_verification = kwargs.get('enable_verification', False)
        self.enable_telemetry = kwargs.get('enable_telemetry', False)
        self._max_workers = kwargs.get('n_worker', os.cpu_count() - 1)

        self._main_tasks: list[str] = []
        self._child_tasks: dict[int, list[str]] = {}
        self._workers: dict[int, Process] = {}
        self._enabled = RawValue(c_bool, False)
        self._request_values = None
        self._signature = None
        self._telemetry = None
        self._monitor_value: FactorValueCollector | None = None

        self.lock = Lock()

    def add_monitor(self, monitor: FactorMonitor):
        super().add_monitor(monitor=monitor)

        monitor_id = monitor.monitor_id

        # already enabled
        if self._enabled:
            LOGGER.info(f'Manager {self.__class__.__name__} already started, monitor can only be added into the main process.')
            self._main_tasks.append(monitor_id)
            return

            # non-concurrent mode
        if self._max_workers <= 1:
            LOGGER.info(f'{self.__class__.__name__} in single-process mode! Assigning monitor {monitor.name} to main worker.')
            self._main_tasks.append(monitor_id)
            return

        # can not be serialized
        if not monitor.serializable:
            LOGGER.debug(f'Monitor {monitor.name} is marked as not serializable. Assigned to the main process.')
            self._main_tasks.append(monitor_id)
            return

        if (worker_id := sum([len(assignment) for assignment in self._child_tasks.values()]) % self._max_workers) in self._child_tasks:
            self._child_tasks[worker_id].append(monitor_id)
        else:
            self._child_tasks[worker_id] = [monitor_id]

        LOGGER.info(f'Assign monitor {monitor.name} to worker {worker_id}.')

    def pop_monitor(self, monitor_id: str):
        if monitor_id in self._main_tasks:
            self._main_tasks.remove(monitor_id)
            LOGGER.debug(f'Monitor {monitor_id} removed from main process.')

        for worker_id, tasks in self._child_tasks.items():
            if monitor_id in tasks:
                tasks.remove(monitor_id)
                LOGGER.debug(f'Monitor {monitor_id} removed from worker {worker_id}.')

        return super().pop_monitor(monitor_id=monitor_id)

    def __call__(self, market_data: MarketData):
        if not self._workers:
            return super().__call__(market_data=market_data)

        if not self.enabled:
            raise ValueError(f'{self.__class__.__name__} is not enabled, please start it first!')

        self._distribute_market_data(market_data=market_data)

        if self.enable_verification:
            if not hasattr(self, '_md5'):
                self._md5 = hashlib.md5()

            self._md5.update(pickle.dumps(market_data))

    @abc.abstractmethod
    def _distribute_market_data(self, market_data: MarketData):
        ...

    @abc.abstractmethod
    def _collect_worker(self, worker_id: int):
        ...

    def _join_worker(self, worker_id: int):
        worker = self._workers[worker_id]
        worker.terminate()
        worker.join()
        worker.close()

    @abc.abstractmethod
    def worker(self, worker_id: int):
        ...

    def _initialize_buffer(self, n_workers: int = None):
        n_workers = len(self._child_tasks) if n_workers is None else n_workers
        self._request_values = RawArray(c_bool, n_workers)
        self._monitor_value = FactorValueCollector()
        self._signature = RawArray(self.Signature, n_workers)
        self._telemetry = RawArray(self.Telemetries, n_workers)

    def start(self):
        if self.enabled:
            raise ValueError(f'{self.__class__.__name__} already started!')

        self.enabled = True
        n_workers = len(self._child_tasks)

        if self._max_workers <= 1:
            LOGGER.info(f'{self.__class__.__name__} in single process mode, and will not spawn any child process.')
            return

        if n_workers <= 2:
            LOGGER.info(f'{self.__class__.__name__} too little child worker, concurrency will not cover cost of overhead, starting in single process mode.')
            return

        # main_tasks = self._main_tasks
        # child_tasks = self._child_tasks
        monitor = self.monitor.copy()
        LOGGER.info(f'{self.__class__.__name__} initializing buffer for {len(self._child_tasks)} workers...')
        self._initialize_buffer(n_workers=len(self._child_tasks))

        workers = {}
        for worker_id, tasks in self._child_tasks.items():
            self.monitor.clear()

            for monitor_id in tasks:
                self.monitor[monitor_id] = monitor[monitor_id]

            if not self.monitor:
                LOGGER.warning(f'No task for worker {worker_id}! Consider reduce the number of the workers!')
                # continue

            p = workers[worker_id] = Process(target=self.worker, name=f'{self.__class__.__name__}.worker.{worker_id}', kwargs={'worker_id': worker_id})
            p.start()

        self._workers.update(workers)
        self.monitor.update(monitor)

        LOGGER.info(f'{self.__class__.__name__} {len(self._workers)} workers of started!')

    def stop(self):
        if not self.enabled:
            LOGGER.info(f'{self.__class__.__name__} already stopped!')
            return

        self.enabled = False

        for worker_id in self._workers:
            self._join_worker(worker_id=worker_id)

        self._workers.clear()

        self._request_values = None
        LOGGER.info(f'{self.__class__.__name__} all worker stopped and cleared.')

    def clear(self):
        # clear shm
        for monitor_id in list(self.monitor):
            self.pop_monitor(monitor_id)

        super().clear()

    def verify_signature(self) -> bool:
        if hasattr(self, '_md5'):
            signature = self._md5.digest()
        else:
            signature = b'\x00' * 16

        for _signature in self._signature:
            if _signature.md5 != signature:
                return False

        return True

    @property
    def enabled(self):
        return self._enabled.value

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled.value = c_bool(value)

    @property
    def values(self) -> dict[str, float]:
        if not self._workers:
            return collect_factor(self.monitor)

        self._request_values.value = True
        monitor_value = collect_factor([self.monitor[monitor_id] for monitor_id in self._main_tasks])

        for worker_id in self._workers:
            self._collect_worker(worker_id=worker_id)

        self._request_values.value = False
        monitor_value.update(self._monitor_value)
        return monitor_value


class AsyncMonitorManager(ConcurrentMonitorManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._buffer = None
        self._update_ready = None
        self._update_barrier = None

    def _initialize_buffer(self, n_workers: int = None):
        n_workers = len(self._child_tasks) if n_workers is None else n_workers

        super()._initialize_buffer(n_workers=n_workers)

        self._buffer = MarketDataBuffer(n_workers=n_workers, size=1 * 1024 * 1024, block=False)
        self._update_ready = RawArray(c_bool, n_workers)
        self._update_barrier = Barrier(n_workers + 1)

    def clear(self):
        super().clear()

        self._buffer = None
        self._update_ready = None
        self._update_barrier = None

    def _distribute_market_data(self, market_data: MarketData):
        self._buffer.put(market_data=market_data)

        for monitor_id in self._main_tasks:
            self._work(monitor_id=monitor_id, market_data=market_data)

    def _collect_worker(self, worker_id: int):
        pass

    def _join_worker(self, worker_id: int):
        if self._buffer.block:
            with self._buffer.condition_get:
                self._buffer.condition_get.notify_all()

        super()._join_worker(worker_id)

    def worker(self, worker_id: int):
        signature = hashlib.md5()

        while True:
            if not self.enabled:
                LOGGER.info(f'Stopping worker {worker_id}...')
                LOGGER.info(f'Worker {worker_id} shm closed! Ready to join the process!')
                break

            # when requesting monitor value, the market data will not be updated.
            if self._request_values[worker_id] and self._update_ready[worker_id]:
                updated_value = collect_factor(self.monitor)

                if self._buffer.head[worker_id] < self._buffer.tail:
                    LOGGER.warning(f'Worker {worker_id} has pending data to process! Factor Value collection might be obsolete!')

                self._request_values[worker_id] = False

                if updated_value:
                    with self.lock:
                        self._monitor_value.update(updated_value)

                self._update_barrier.wait()
            else:
                market_data = self._buffer.get(worker_id=worker_id)

                if market_data is None:
                    if self._request_values[worker_id]:
                        self._update_ready[worker_id] = True
                    continue

                for monitor_id in self.monitor:
                    self._work(monitor_id=monitor_id, market_data=market_data)

                if self.enable_verification:
                    signature.update(pickle.dumps(market_data))
                    self._signature[worker_id].md5 = signature.digest()

    @property
    def values(self) -> dict[str, float]:
        if not self._workers:
            return collect_factor(self.monitor)

        # the put process should be locked by default, this is the same thread feeding the buffer
        monitor_value = collect_factor([self.monitor[monitor_id] for monitor_id in self._main_tasks])

        while not self._buffer.is_empty():
            continue

        # _request_values flag will be reset by each worker.
        memset(self._request_values, True, len(self._workers))

        while not all(self._update_ready):
            continue

        if self._buffer.block:
            with self._buffer.condition_get:
                self._buffer.condition_get.notify_all()

        if self.enable_verification:
            assert self.verify_signature(), f'Market Data chain verification failed! Processes not synchronized! Hashes as follows:\n{"\n".join([str(_.md5) for _ in self._signature])}'

        self._update_barrier.wait()
        self._update_barrier.reset()
        monitor_value.update(self._monitor_value)
        memset(self._update_ready, False, len(self._workers))

        return monitor_value


class SyncMonitorManager(ConcurrentMonitorManager):
    def __init__(self, n_worker: int):
        super().__init__(n_worker=n_worker)

        self.request_value = RawValue(c_bool, False)
        self.lock = Lock()
        self.semaphore_start = Semaphore(value=0)
        self.semaphore_done = Semaphore(value=0)

    def _initialize_buffer(self, n_workers: int = None):
        n_workers = len(self._child_tasks) if n_workers is None else n_workers

        super()._initialize_buffer(n_workers=n_workers)

        from algo_engine.base import MarketDataBuffer as _Buffer
        self.buffer = _Buffer()
        self.request_value = RawValue(c_bool, False)

    def _distribute_market_data(self, market_data: MarketData):
        # step 1: send market data to the shared memory
        self.buffer.update(market_data=market_data)

        # step 2: release the semaphore
        for worker_id in self._workers:
            self.semaphore_start.release()

        # step 3: execute tasks in main thread for those monitors not supporting multiprocessing features
        for monitor_id in self._main_tasks:
            self._work(monitor_id=monitor_id, market_data=market_data)

        # step 4: acquire semaphore to wait till the tasks all done
        for worker_id in self._workers:
            self.semaphore_done.acquire()

    def _collect_worker(self, worker_id: int):
        pass

    def _join_worker(self, worker_id: int):
        for _ in self._workers:
            self.semaphore_start.release()

        super()._join_worker(worker_id)

    def worker(self, worker_id: int):
        while True:
            self.semaphore_start.acquire()

            # management job 0: terminate the worker on signal
            if not self.enabled:
                break

            # when requesting monitor value, the market data will not be updated.
            if self.request_value.value:
                updated_value = collect_factor(self.monitor)

                if updated_value:
                    with self.lock:
                        self._monitor_value.update(updated_value)
            else:
                # step 1.1: reconstruct market data
                market_data = self.buffer.contents

                # step 2: do the tasks
                for monitor_id in self.monitor:
                    self._work(monitor_id=monitor_id, market_data=market_data)

            self.semaphore_done.release()

    def clear(self):
        # clear shm
        for monitor_id in list(self.monitor):
            self.pop_monitor(monitor_id)

        super().clear()

        while self.semaphore_start.acquire(False):
            continue

        while self.semaphore_done.acquire(False):
            continue
