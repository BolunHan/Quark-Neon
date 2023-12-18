from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData

from .. import Synthetic, LOGGER


class MACDMonitor(MarketDataMonitor, Synthetic):
    """
    as the name suggest, it gives a realtime macd value
    """

    class MACD:
        def __init__(self, short_window=12, long_window=26, signal_window=9):
            self.short_window = short_window
            self.long_window = long_window
            self.signal_window = signal_window

            self.ema_short = None
            self.ema_long = None

            self.macd_line = None
            self.signal_line = None
            self.macd_histogram = None
            self.close_price = None
            self.timestamp = None

        @classmethod
        def update_ema(cls, close_price: float, ema_prev: float, window: int):
            alpha = 2 / (window + 1)
            ema = alpha * close_price + (1 - alpha) * ema_prev
            return ema

        def update_macd(self, close_price: float):
            if self.ema_short is None:
                # Initialize EMA values
                self.ema_short = close_price
                self.ema_long = close_price
            else:
                # Update EMA values
                self.ema_short = self.update_ema(close_price, self.ema_short, self.short_window)
                self.ema_long = self.update_ema(close_price, self.ema_long, self.long_window)

            # Calculate MACD line
            self.macd_line = self.ema_short - self.ema_long

            if self.signal_line is None:
                # Initialize signal line
                self.signal_line = self.macd_line
            else:
                # Update signal line
                self.signal_line = self.update_ema(self.macd_line, self.signal_line, self.signal_window)

            # Calculate MACD histogram
            self.macd_histogram = self.macd_line - self.signal_line
            self.close_price = close_price

        def get_macd_values(self):
            return self.macd_line, self.signal_line, self.macd_histogram

    def __init__(self, update_interval: float, weights: dict[str, float], name: str = 'Monitor.TA.MACD', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

        self.update_interval = update_interval

        self._macd: dict[str, MACDMonitor.MACD] = {}
        self._price = {}
        self._is_ready = True

        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        if ticker in self._macd:
            macd = self._macd[ticker]
        else:
            macd = self._macd[ticker] = self.MACD()

        last_update = macd.timestamp

        if last_update is None or last_update + self.update_interval < timestamp:
            macd.update_macd(close_price=self._price.get(ticker, market_price))
            macd.timestamp = timestamp // self.update_interval * self.update_interval

        self._price[ticker] = market_price

    @property
    def value(self) -> dict[str, float]:
        macd_value = {}

        for ticker in self._macd:
            macd_value[ticker] = self._macd[ticker].macd_histogram

        return macd_value

    @property
    def index_value(self) -> float:
        return self.composite(self.value)

    @property
    def is_ready(self) -> bool:
        return self._is_ready
