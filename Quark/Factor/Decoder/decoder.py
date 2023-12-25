from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData

from .. import MDS, Synthetic
from ..decoder import OnlineDecoder, Wavelet


class DecoderMonitor(MarketDataMonitor, OnlineDecoder):
    """
    mark the market movement into different trend:
    - when price goes up to 1% (up_threshold)
    - when price goes down to 1% (down_threshold)
    - when price goes up / down, relative to the local minimum / maximum, more than 0.5%  (confirmation_level)
    - when the market trend goes on more than 15 * 60 seconds (timeout)

    upon a new trend is confirmed, a callback function will be called.
    """

    def __init__(self, confirmation_level: float = 0.005, timeout: float = 15 * 60, up_threshold: float = 0.01, down_threshold: float = 0.01, retrospective: bool = False, name: str = 'Monitor.Decoder', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        OnlineDecoder.__init__(self=self, confirmation_level=confirmation_level, timeout=timeout, up_threshold=up_threshold, down_threshold=down_threshold, retrospective=retrospective)

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self.update_decoder(ticker=ticker, market_price=market_price, timestamp=timestamp)

    def clear(self):
        OnlineDecoder.clear(self)

    @property
    def value(self) -> dict[str, list[Wavelet]]:
        return self.state_history

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class IndexDecoderMonitor(DecoderMonitor, Synthetic):
    def __init__(self, weights: dict[str, float], confirmation_level: float = 0.005, timeout: float = 15 * 60, up_threshold: float = 0.01, down_threshold: float = 0.01, retrospective: bool = False, name: str = 'Monitor.Decoder.Index', monitor_id: str = None):
        super().__init__(confirmation_level=confirmation_level, timeout=timeout, up_threshold=up_threshold, down_threshold=down_threshold, retrospective=retrospective, name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        self._update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)
        self.update_decoder(ticker='synthetic', market_price=self.synthetic_index, timestamp=market_data.timestamp)

    def clear(self):
        super().clear()
        Synthetic.clear(self)

    @property
    def value(self) -> list[Wavelet]:
        return self.state_history['synthetic']
