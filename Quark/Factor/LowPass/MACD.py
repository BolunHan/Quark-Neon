from collections import deque

from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData

from .. import Synthetic, LOGGER, MDS


class MACD(object):
    """
    This model calculates the MACD absolute value (not the relative / adjusted value)

    use update_macd method to update the close price
    """

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


class MACDMonitor(MarketDataMonitor, Synthetic):
    """
    as the name suggest, it gives a realtime macd value
    """

    def __init__(self, update_interval: float, weights: dict[str, float], name: str = 'Monitor.TA.MACD', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

        self.update_interval = update_interval

        self._macd: dict[str, MACD] = {}
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
            macd = self._macd[ticker] = MACD()

        last_update = macd.timestamp

        if last_update is None or last_update + self.update_interval < timestamp:
            macd.update_macd(close_price=self._price.get(ticker, market_price))
            macd.timestamp = timestamp // self.update_interval * self.update_interval

        self._price[ticker] = market_price

    def clear(self):
        self._macd.clear()
        self._price.clear()
        Synthetic.clear(self)

    def macd_value(self):
        macd_value = {}

        for ticker in self._macd:
            macd_value[ticker] = self._macd[ticker].macd_histogram

        return macd_value

    @property
    def value(self) -> dict[str, float]:
        result = {}
        result.update(self.macd_value())
        result['Index'] = self.index_value
        return result

    @property
    def index_value(self) -> float:
        return self.composite(self.macd_value())

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class MACDTriggerMonitor(MarketDataMonitor):
    """
    given an <update_interval>, this monitor logs #<observation_windows> observation

    the monitor value is defined as :
    in the logged observation:
        if min of adjusted macd < -<confirmation_threshold>, but the latest value > 0 => 1
        if max of adjusted macd > <confirmation_threshold>, but the latest value < 0 => -1
        else 0

    this monitor is designed to signal the event of a MACD sign change
    the confirmation threshold is designed to filter out fluctuation.
    """

    def __init__(self, update_interval: float, observation_window: int, confirmation_threshold: float, name: str = 'Monitor.MACD.Trigger', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

        self.update_interval = update_interval
        self.observation_window = observation_window
        self.confirmation_threshold = confirmation_threshold

        self._macd: dict[str, MACD] = {}
        self._macd_adjusted: dict[str, deque[float]] = {}
        self._macd_last_extreme: dict[str, dict[str, float]] = {}
        self._price: dict[str, float] = {}
        self._is_ready = True

        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

    def __call__(self, market_data: MarketData, **kwargs):
        self._update_macd(
            ticker=market_data.ticker,
            market_price=market_data.market_price,
            timestamp=market_data.timestamp
        )

    def _update_macd(self, ticker: str, market_price: float, timestamp: float):

        if ticker in self._macd:
            macd = self._macd[ticker]
            macd_storage = self._macd_adjusted[ticker]
            macd_extreme = self._macd_last_extreme[ticker]
        else:
            macd = self._macd[ticker] = MACD()
            macd_storage = self._macd_adjusted[ticker] = deque(maxlen=self.observation_window)
            macd_extreme = self._macd_last_extreme[ticker] = {}

        last_update = macd.timestamp

        if last_update is None or last_update + self.update_interval < timestamp:
            close_price = self._price.get(ticker, market_price)
            macd.update_macd(close_price=close_price)
            macd.timestamp = timestamp // self.update_interval * self.update_interval
            last_macd = macd_storage[-1] if macd_storage else 0.
            macd_adjusted = macd.macd_histogram / close_price
            macd_storage.append(macd_adjusted)

            if last_macd <= 0 < macd_adjusted:
                macd_extreme['max'] = macd_adjusted
            elif last_macd >= 0 > macd_adjusted:
                macd_extreme['min'] = macd_adjusted
            elif macd_adjusted > 0:
                macd_extreme['max'] = max(macd_adjusted, macd_extreme.get('max', 0.))
            elif macd_adjusted < 0:
                macd_extreme['min'] = min(macd_adjusted, macd_extreme.get('min', 0.))
            else:
                macd_extreme['min'] = 0.
                macd_extreme['max'] = 0.

        self._price[ticker] = market_price

    def clear(self):
        self._macd.clear()
        self._price.clear()
        self._macd_adjusted.clear()

    @property
    def value(self) -> dict[str, float]:
        monitor_value = {}

        for ticker in self._macd_adjusted:
            storage = self._macd_adjusted[ticker]
            last_extreme = self._macd_last_extreme[ticker]

            if storage:
                if (
                        min(storage) <= 0. and
                        last_extreme.get('min', 0.) <= -self.confirmation_threshold and
                        storage[-1] > 0
                ):
                    state = 1.
                elif (
                        max(storage) >= 0. and
                        last_extreme.get('max', 0.) >= self.confirmation_threshold and
                        storage[-1] < 0
                ):
                    state = -1.
                else:
                    state = 0.

                monitor_value[ticker] = state

        return monitor_value

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class IndexMACDTriggerMonitor(MACDTriggerMonitor, Synthetic):

    def __init__(self, weights: dict[str, float], update_interval: float, observation_window: int, confirmation_threshold: float, name: str = 'Monitor.MACD.Index.Trigger', monitor_id: str = None):
        super().__init__(update_interval=update_interval, observation_window=observation_window, confirmation_threshold=confirmation_threshold, name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        super().__call__(market_data=market_data, **kwargs)
        self._update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)
        self._update_macd(ticker='Synthetic', market_price=self.synthetic_index, timestamp=market_data.timestamp)

    def clear(self):
        super().clear()
        Synthetic.clear(self)
