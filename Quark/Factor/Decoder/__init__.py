from .volatility import VolatilityMonitor
from .decoder import DecoderMonitor, IndexDecoderMonitor
from .. import add_monitor, INDEX_WEIGHTS

MONITOR = {}

add_monitor(VolatilityMonitor(weights=INDEX_WEIGHTS))
add_monitor(DecoderMonitor(retrospective=False))
add_monitor(IndexDecoderMonitor(up_threshold=0.005, down_threshold=0.005, confirmation_level=0.002, retrospective=True, weights=INDEX_WEIGHTS))
