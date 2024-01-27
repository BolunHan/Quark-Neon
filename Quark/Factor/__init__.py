import abc
from functools import partial
from typing import Iterable

from AlgoEngine.Engine import MarketDataMonitor, MDS

from .. import LOGGER
from ..Base import GlobalStatics

LOGGER = LOGGER.getChild('Factor')
TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE


class FactorMonitor(MarketDataMonitor, metaclass=abc.ABCMeta):
    def __init__(self, name: str, monitor_id: str = None):
        assert name.startswith('Monitor')
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

    @abc.abstractmethod
    def factor_names(self, subscription: list[str]) -> list[str]: ...

    """
    This method returns a list of string, corresponding with the keys of the what .value returns.
    This method is design to facilitate facter caching functions.
    """


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
    'FactorMonitor', 'LOGGER', 'DEBUG_MODE', 'register_monitor', 'IndexWeight', 'Synthetic', 'EMA', 'register_monitor', 'collect_factor',
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
    'AggressivenessMonitor', 'AggressivenessEMAMonitor', 'TradeFlowMonitor', 'TradeFlowEMAMonitor',
]
