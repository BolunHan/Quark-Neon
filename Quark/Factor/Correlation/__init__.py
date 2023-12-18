from .coherence import CoherenceMonitor, CoherenceEMAMonitor, TradeCoherenceMonitor
from .entropy import EntropyMonitor, EntropyEMAMonitor
from .. import add_monitor, ALPHA_0001, INDEX_WEIGHTS

MONITOR = {}

add_monitor(CoherenceMonitor(update_interval=60, sample_interval=1, weights=INDEX_WEIGHTS))
add_monitor(TradeCoherenceMonitor(update_interval=60, sample_interval=1, weights=INDEX_WEIGHTS))
add_monitor(CoherenceEMAMonitor(update_interval=60, sample_interval=1, weights=INDEX_WEIGHTS, discount_interval=1, alpha=ALPHA_0001))

add_monitor(EntropyMonitor(update_interval=60, sample_interval=1, weights=INDEX_WEIGHTS))
add_monitor(EntropyEMAMonitor(update_interval=60, sample_interval=1, weights=INDEX_WEIGHTS, discount_interval=1, alpha=ALPHA_0001))
