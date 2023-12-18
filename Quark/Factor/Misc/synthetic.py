import datetime

from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData, BarData

from .. import Synthetic, MDS, TIME_ZONE


class SyntheticIndexMonitor(MarketDataMonitor, Synthetic):
    """
    a monitor to synthesize the index price / volume movement
    """

    def __init__(self, index_name: str, weights: dict[str, float], interval: float = 60., name='Monitor.SyntheticIndex', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS, )
        Synthetic.__init__(self=self, weights=weights)

        self.index_name = index_name
        self.interval = interval

        self._active_bar_data: BarData | None = None
        self._last_bar_data: BarData | None = None

        self._is_ready = True
        self._value = {}

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        self._update_synthetic(ticker=ticker, market_price=market_price)

        if ticker not in self.weights:
            return

        index_price = self.synthetic_index

        if self._active_bar_data is None or timestamp >= self._active_bar_data.timestamp:
            self._last_bar_data = self._active_bar_data
            bar_data = self._active_bar_data = BarData(
                ticker=self.index_name,
                bar_start_time=datetime.datetime.fromtimestamp(timestamp // self.interval * self.interval, tz=TIME_ZONE),
                timestamp=(timestamp // self.interval + 1) * self.interval,  # by definition, timestamp when bar end
                bar_span=datetime.timedelta(seconds=self.interval),
                high_price=index_price,
                low_price=index_price,
                open_price=index_price,
                close_price=index_price,
                volume=0.,
                notional=0.,
                trade_count=0
            )
        else:
            bar_data = self._active_bar_data

        if isinstance(market_data, TradeData):
            bar_data.volume += market_data.volume
            bar_data.notional += market_data.notional
            bar_data.trade_count += 1

        bar_data.close_price = index_price
        bar_data.high_price = max(bar_data.high_price, index_price)
        bar_data.low_price = min(bar_data.low_price, index_price)

    @property
    def is_ready(self) -> bool:
        if self._last_bar_data is None:
            return False
        else:
            return self._is_ready

    @property
    def value(self) -> BarData | None:
        return self._last_bar_data

    @property
    def index_price(self) -> float:
        return self.synthetic_index

    @property
    def active_bar(self):
        return self._active_bar_data
