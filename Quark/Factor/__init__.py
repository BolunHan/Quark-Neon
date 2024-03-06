from __future__ import annotations

import os
from typing import Iterable

from AlgoEngine.Engine import MarketDataMonitor, MDS

from .. import LOGGER
from ..Base import GlobalStatics

LOGGER = LOGGER.getChild('Factor')
TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE
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
            if DEBUG_MODE and monitor.serializable and not monitor.is_sync:
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


from .utils import *

if ENABLE_SHM:  # mask the singleton codes
    from .utils_shm import *
from .ta import *

INDEX_WEIGHTS = IndexWeight(index_name='DummyIndex')
MONITOR_MANAGER = ConcurrentMonitorManager(n_worker=N_CORES)

from .TradeFlow import *
from .Misc import *


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
    'FactorMonitor', 'LOGGER', 'DEBUG_MODE', 'IndexWeight', 'MONITOR_MANAGER', 'Synthetic', 'EMA', 'collect_factor', 'N_CORES',
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
