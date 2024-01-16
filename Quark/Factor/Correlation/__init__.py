from .coherence import CoherenceMonitor, CoherenceEMAMonitor, TradeCoherenceMonitor
from .entropy import EntropyMonitor, EntropyAdaptiveMonitor, EntropyEMAMonitor
from .. import add_monitor, ALPHA_0001, INDEX_WEIGHTS

MONITOR = {}

add_monitor(CoherenceMonitor(sampling_interval=15, sample_size=20, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
add_monitor(TradeCoherenceMonitor(sampling_interval=15, sample_size=20, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
add_monitor(CoherenceEMAMonitor(sampling_interval=15, sample_size=20, weights=INDEX_WEIGHTS, discount_interval=1, alpha=ALPHA_0001), monitors=MONITOR, register=False)

add_monitor(EntropyMonitor(sampling_interval=15, sample_size=20, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
add_monitor(EntropyEMAMonitor(sampling_interval=15, sample_size=20, weights=INDEX_WEIGHTS, discount_interval=1, alpha=ALPHA_0001), monitors=MONITOR, register=False)

add_monitor(EntropyAdaptiveMonitor(sampling_interval=15, sample_size=20, pct_change=False, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
__all__ = ['CoherenceMonitor', 'CoherenceEMAMonitor', 'TradeCoherenceMonitor',
           'EntropyMonitor', 'EntropyAdaptiveMonitor', 'EntropyEMAMonitor']
