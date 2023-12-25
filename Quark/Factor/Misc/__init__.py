from .synthetic import SyntheticIndexMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(SyntheticIndexMonitor(index_name=INDEX_WEIGHTS.index_name, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
