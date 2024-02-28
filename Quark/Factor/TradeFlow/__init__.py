from .aggressiveness import AggressivenessMonitor, AggressivenessEMAMonitor
from .trade_flow import TradeFlowMonitor, TradeFlowAdaptiveMonitor, TradeFlowAdaptiveIndexMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(TradeFlowMonitor(sampling_interval=15, sample_size=20), monitors=MONITOR, register=False)
add_monitor(TradeFlowAdaptiveIndexMonitor(sampling_interval=15, sample_size=20, baseline_window=100, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)

add_monitor(AggressivenessMonitor(), monitors=MONITOR, register=False)

__all__ = ['AggressivenessMonitor',
           'TradeFlowMonitor', 'TradeFlowAdaptiveMonitor', 'TradeFlowAdaptiveIndexMonitor']
