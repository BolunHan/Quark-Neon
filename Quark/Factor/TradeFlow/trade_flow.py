from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData

from .. import EMA, Synthetic, MDS


class TradeFlowMonitor(MarketDataMonitor):
    """
    monitor net trade volume flow of given underlying, with unit of share
    """

    def __init__(self, name: str = 'Monitor.TradeFlow', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

        self._trade_flow = dict()
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, TradeData):
            return self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign

        self._trade_flow[ticker] = self._trade_flow.get(ticker, 0.) + volume * side

    def clear(self):
        self._trade_flow.clear()

    @property
    def value(self) -> dict[str, float]:
        return self._trade_flow

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class TradeFlowEMAMonitor(TradeFlowMonitor, EMA, Synthetic):
    """
    ema discounted trade flow
    Note that the discount process is triggered by on_trade_data. Upon discount, there should be a discontinuity of index value.
    But if the update interval is small enough, the effect should be marginal

    trade flow is a relative stable indication. a large positive aggressiveness indicates market is in an upward ongoing trend
    """

    def __init__(self, discount_interval: float, alpha: float, weights: dict[str, float], normalized: bool = True, name: str = 'Monitor.TradeFlow.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)
        Synthetic.__init__(self=self, weights=weights)

        self.normalized = normalized

        self._trade_flow = self._register_ema(name='trade_flow')
        self._trade_volume = self._register_ema(name='trade_volume')

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        self._accumulate_ema(ticker=ticker, timestamp=timestamp, trade_flow=volume * side, trade_volume=volume)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker=ticker, timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

    def clear(self):
        super().clear()
        EMA.clear(self)
        Synthetic.clear(self)

    @property
    def value(self) -> dict[str, float]:
        if self.normalized:
            normalized_trade_flow = {}

            for ticker in self._trade_flow:
                net_flow = self._trade_flow[ticker]
                volume = self._trade_volume[ticker]
                adjusted_flow = net_flow / volume if volume else 0.

                normalized_trade_flow[ticker] = adjusted_flow

                if abs(adjusted_flow) > 1:
                    raise ValueError('adjusted_flow should not larger than 1 or smaller than -1, check the code!')

            return normalized_trade_flow
        else:
            return super().value

    @property
    def index_value(self) -> float:
        return self.composite(self.value)
