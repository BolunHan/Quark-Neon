import logging
import os
from typing import Iterable

from algo_engine.engine import MarketDataMonitor, MDS

from .. import LOGGER
from ..base import GlobalStatics

LOGGER = LOGGER.getChild('Factor')
# to disable multiprocessing, set this value to 1
# N_CORES = 1
N_CORES = os.cpu_count()
ENABLE_SHM = False


def collect_factor(monitors: dict[str, MarketDataMonitor] | list[MarketDataMonitor] | MarketDataMonitor) -> dict[str, float]:
    factors = {}

    if isinstance(monitors, dict):
        monitors = list(monitors.values())
    elif isinstance(monitors, FactorMonitor):
        monitors = [monitors]

    for monitor in monitors:
        if monitor.is_ready and monitor.enabled:
            if GlobalStatics.DEBUG_MODE and monitor.serializable and not monitor.is_sync:
                monitor.memory_core.from_shm()

            factor_value = monitor.value
            name = monitor.name.removeprefix('Monitor.')

            if isinstance(factor_value, (int, float)):
                factors[name] = factor_value
            elif isinstance(factor_value, dict):
                # FactorPoolDummyMonitor having hard coded name
                if monitor.name == 'Monitor.FactorPool.Dummy':
                    factors.update(factor_value)
                # synthetic index monitor should have duplicated logs
                elif monitor.__class__.__name__ == 'SyntheticIndexMonitor':
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
                    factors.update({f'{monitor.index_name}.{key}': value for key, value in factor_value.items()})
                else:
                    factors.update({f'{name}.{key}': value for key, value in factor_value.items()})
            else:
                raise NotImplementedError(f'Invalid return type, expect float | dict[str, float], got {type(factor_value)}.')

    return factors


def set_logger(logger: logging.Logger):
    from . import decoder
    from . import factor_pool
    from . import memory_core
    from . import utils, utils_shm

    decoder.LOGGER = logger.getChild('Decoder')
    factor_pool.LOGGER = LOGGER.getChild('FactorPool')
    memory_core.LOGGER = LOGGER.getChild('MemoryCore')
    utils.LOGGER = LOGGER.getChild('Utils')
    utils_shm.LOGGER = LOGGER.getChild('Utils.SHM')


if ENABLE_SHM:  # mask the singleton codes
    from .utils_shm import *
else:
    from .utils import *

from .ta import *

INDEX_WEIGHTS = IndexWeight(index_name='DummyIndex')
# MONITOR_MANAGER = ConcurrentMonitorManager(n_worker=N_CORES)

from .trade_flow import *
from .misc import *


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


__all__ = [
    'FactorMonitor', 'IndexWeight', 'Synthetic', 'EMA', 'collect_factor', 'N_CORES',
    # from misc module
    'SyntheticIndexMonitor',
    # from utils module
    'FactorMonitor', 'ConcurrentMonitorManager', 'EMA',
    'ALPHA_05', 'ALPHA_02', 'ALPHA_01', 'ALPHA_001', 'ALPHA_0001',
    'Synthetic', 'IndexWeight',
    'SamplerMode', 'FixedIntervalSampler', 'FixedVolumeIntervalSampler', 'AdaptiveVolumeIntervalSampler'
]
