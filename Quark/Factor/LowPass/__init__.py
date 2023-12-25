from .MACD import MACDMonitor, MACDTriggerMonitor

from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(MACDMonitor(weights=INDEX_WEIGHTS, update_interval=60), monitors=MONITOR, register=False)
add_monitor(MACDTriggerMonitor(update_interval=60, observation_window=5, confirmation_threshold=0.0001), monitors=MONITOR, register=False)
