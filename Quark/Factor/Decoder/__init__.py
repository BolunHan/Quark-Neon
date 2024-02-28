from .decoder import DecoderMonitor, IndexDecoderMonitor
from .volatility import VolatilityMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(VolatilityMonitor(weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)
add_monitor(DecoderMonitor(retrospective=False), monitors=MONITOR, register=False)
add_monitor(IndexDecoderMonitor(up_threshold=0.005, down_threshold=0.005, confirmation_level=0.002, retrospective=True, weights=INDEX_WEIGHTS), monitors=MONITOR, register=False)

__all__ = ['VolatilityMonitor', 'DecoderMonitor', 'IndexDecoderMonitor']
