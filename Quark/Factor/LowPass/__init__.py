from .clustering import TradeClusteringMonitor, TradeClusteringAdaptiveMonitor, TradeClusteringIndexAdaptiveMonitor
from .divergence import DivergenceMonitor, DivergenceAdaptiveMonitor, DivergenceIndexAdaptiveMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(DivergenceMonitor(sampling_interval=15), monitors=MONITOR, register=False)
add_monitor(DivergenceAdaptiveMonitor(sampling_interval=15, baseline_window=15, aligned_interval=False), monitors=MONITOR, register=False)
add_monitor(DivergenceIndexAdaptiveMonitor(weights=INDEX_WEIGHTS, sampling_interval=15, baseline_window=15, aligned_interval=False), monitors=MONITOR, register=False)

__all__ = ['TradeClusteringMonitor', 'TradeClusteringAdaptiveMonitor', 'TradeClusteringIndexAdaptiveMonitor',
           'DivergenceMonitor', 'DivergenceAdaptiveMonitor', 'DivergenceIndexAdaptiveMonitor']
