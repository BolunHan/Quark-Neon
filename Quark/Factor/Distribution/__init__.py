from .moments import SkewnessMonitor, SkewnessIndexMonitor, SkewnessAdaptiveMonitor, SkewnessIndexAdaptiveMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(SkewnessIndexAdaptiveMonitor(sampling_interval=3 * 5, sample_size=20, baseline_window=100, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)

__all__ = ['SkewnessMonitor', 'SkewnessIndexMonitor', 'SkewnessAdaptiveMonitor', 'SkewnessIndexAdaptiveMonitor']
