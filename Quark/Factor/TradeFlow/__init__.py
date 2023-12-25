from .aggressiveness import AggressivenessMonitor, AggressivenessEMAMonitor
from .trade_flow import TradeFlowMonitor, TradeFlowEMAMonitor
from .. import add_monitor, ALPHA_05, ALPHA_0001, INDEX_WEIGHTS

MONITOR = {}

add_monitor(TradeFlowMonitor(), monitors=MONITOR, register=False)
add_monitor(TradeFlowEMAMonitor(discount_interval=1, alpha=ALPHA_05, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)

add_monitor(AggressivenessMonitor(), monitors=MONITOR, register=False)
add_monitor(AggressivenessEMAMonitor(discount_interval=1, alpha=ALPHA_0001, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
