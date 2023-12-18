from .MACD import MACDMonitor

from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(MACDMonitor(weights=INDEX_WEIGHTS, update_interval=60))
