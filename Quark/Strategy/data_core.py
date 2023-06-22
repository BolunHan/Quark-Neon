"""
implement MarketDataMonitor and add it to the register() method
"""

import datetime

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData, BarData

from . import LOGGER, MDS


class TradeFlowMonitor(MarketDataMonitor):
    """
    monitor net trade volume flow of given underlying, with unit of share
    """

    def __init__(self, name: str = 'Monitor.TradeFlow', monitor_id: str = None):
        super().__init__(
            name=name,
            monitor_id=monitor_id,
            mds=MDS
        )

        self._trade_flow = dict()
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        if not isinstance(market_data, TradeData):
            return

        ticker = market_data.ticker
        volume = market_data.volume
        side = market_data.side.sign

        self._trade_flow[ticker] = self._trade_flow.get(ticker, 0.) + volume * side

    def clear(self):
        self._trade_flow.clear()

    @property
    def value(self) -> dict[str, float]:
        return self._trade_flow

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class TradeFlowEMAMonitor(TradeFlowMonitor):
    """
    ema discounted trade flow
    Note that the discount process is triggered by on_trade_data. Upon discount, there should be a discontinuity of index value.
    But if the update interval is small enough, the effect should be marginal
    """

    def __init__(self, update_interval: float, alpha: float, name: str = 'Monitor.TradeFlow.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

        self.update_interval = update_interval
        self.alpha = alpha
        self._last_discounted_update = dict()

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.name} should have an alpha from 0 to 1')

        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

    def _discount_trade_flow(self, ticker: str, timestamp: float):
        last_update = self._last_discounted_update.get(ticker, 0.)

        if last_update + self.update_interval < timestamp:
            time_span = timestamp - last_update
            adjust_power = time_span // self.update_interval
            self._trade_flow[ticker] = self._trade_flow.get(ticker, 0.) * (self.alpha ** adjust_power)
            self._last_discounted_update[ticker] = timestamp // self.update_interval * self.update_interval

    def _check_discontinuity(self, timestamp: float, tolerance: int = 1):
        discontinued = []

        for ticker in self._last_discounted_update:
            last_update = self._last_discounted_update[ticker]

            if last_update + tolerance * (self.update_interval + 1) < timestamp:
                discontinued.append(ticker)

        return discontinued

    def __call__(self, market_data: MarketData, **kwargs):
        self._discount_trade_flow(ticker=market_data.ticker, timestamp=market_data.timestamp)

        if not isinstance(market_data, TradeData):
            return

        ticker = market_data.ticker
        volume = market_data.volume
        side = market_data.side.sign

        # since the trade data is extremely unlikely to be 2 intervals behind, the power factor is set to either 0 or 1 to save computational power
        alpha = 1 if self._last_discounted_update.get(ticker, 0.) < market_data.timestamp else self.alpha

        self._trade_flow[ticker] = self._trade_flow.get(ticker, 0.) + volume * side * alpha


class PriceCoherenceMonitor(MarketDataMonitor):
    """
    measure the coherence of price pct change
    the basic assumption is that for a stock pool, rank the gaining / losing stock by this price_change_pct

    the rank and log(price_pct) should have a linear correlation.

    the correlation coefficient should give a proximate indication of price change coherence
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price', monitor_id: str = None):
        super().__init__(
            name=name,
            monitor_id=monitor_id,
            mds=MDS
        )

        self.weights = weights
        self.update_interval = update_interval
        self.sample_interval = sample_interval
        self._historical_price: dict[str, dict[float, float]] = {}
        self._price_change_pct: dict[str, float] = {}
        self._is_ready = True

        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

        if update_interval / 2 < sample_interval:
            LOGGER.warning(f"{self.name} should have a smaller sample_interval by Shannon's Theorem, max value {update_interval / 2}")

        if not (update_interval / sample_interval).is_integer():
            LOGGER.error(f"{self.name} should have a smaller sample_interval that is a fraction of the update_interval")

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        sampled_price = self._historical_price.get(ticker, {})

        # update sampled price
        baseline_timestamp = (timestamp // self.update_interval - 1) * self.update_interval
        sample_timestamp = timestamp // self.sample_interval * self.sample_interval
        sampled_price[sample_timestamp] = market_price

        for ts in list(sampled_price):
            if ts < baseline_timestamp:
                sampled_price.pop(ts)
            else:
                # since the sampled_price is always ordered, no need for further check
                break

        # update price pct change
        baseline_price = list(sampled_price.values())[0]  # the sampled_price must be ordered! Guaranteed by  consistency of the order of trade data
        price_change_pct = market_price / baseline_price - 1
        self._price_change_pct[ticker] = price_change_pct

        self._historical_price[ticker] = sampled_price

    @classmethod
    def regression(cls, y: list[float] | np.ndarray, x: list[float] | np.ndarray = None):
        y = np.array(y)
        x = np.array(x)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator

        return slope

    @property
    def value(self) -> tuple[float, float]:
        price_pct_change = []

        if self.weights:
            price_pct_change.extend([self._price_change_pct[ticker] for ticker in self.weights])
        else:
            price_pct_change.extend(self._price_change_pct.values())

        up_list = np.log([_ for _ in price_pct_change if _ > 0])
        down_list = np.log([-_ for _ in price_pct_change if _ < 0])

        up_dispersion = np.nan if len(up_list) < 3 else self.regression(y=up_list, x=np.argsort(up_list).argsort() + 1)
        down_dispersion = np.nan if len(down_list) < 3 else self.regression(y=down_list, x=np.argsort(down_list).argsort() + 1)

        return up_dispersion, down_dispersion

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class SyntheticIndexMonitor(MarketDataMonitor):
    """
    a monitor to synthesize the index price / volume movement
    """

    def __init__(self, index_name: str, weights: dict[str, float], interval: float = 60., name='Monitor.SyntheticIndex', monitor_id: str = None):
        self.index_name = index_name
        self.weights = weights
        self.last_price = {_: np.nan for _ in weights}
        self.interval = interval

        super().__init__(
            name=name,
            monitor_id=monitor_id,
            mds=MDS,
        )

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {_: np.nan for _ in weights}
        self.index_base_price = 1.
        self._active_bar_data: BarData | None = None
        self._last_bar_data: BarData | None = None

        self._is_ready = True
        self._value = {}

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        if ticker not in self.weights:
            return

        if ticker not in self.base_price:
            self.base_price[ticker] = market_price
        self.last_price[ticker] = market_price

        price_list = []
        weight_list = []

        for _ in self.weights:
            weight_list.append(self.weights[_])

            if ticker in self.last_price:
                price_list.append(self.last_price[_] / self.base_price[_])
            else:
                price_list.append(1.)

        index_price = np.average(price_list, weights=weight_list) * self.index_base_price

        if self._active_bar_data is None or timestamp >= self._active_bar_data.timestamp:
            self._last_bar_data = self._active_bar_data
            bar_data = self._active_bar_data = BarData(
                ticker=self.index_name,
                bar_start_time=datetime.datetime.fromtimestamp(timestamp // self.interval * self.interval),
                timestamp=timestamp // self.interval * (self.interval + 1),  # by definition, timestamp when bar end
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
    def active_bar(self):
        return self._active_bar_data


class TradeCoherenceMonitor(PriceCoherenceMonitor):
    """
    similar like the price coherence, the price_change_pct should also have a linear correlation of net trade flow pct
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Volume', monitor_id: str = None):
        super().__init__(
            update_interval=update_interval,
            sample_interval=sample_interval,
            weights=weights,
            name=name,
            monitor_id=monitor_id,
        )

        self._historical_volume: dict[str, dict[float, float]] = {}
        self._historical_volume_net: dict[str, dict[float, float]] = {}

        self._volume_pct: dict[str, float] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        if not isinstance(market_data, TradeData):
            return

        ticker = market_data.ticker
        market_price = market_data.market_price
        volume = market_data.volume
        side = market_data.side.sign
        timestamp = market_data.timestamp

        sampled_price = self._historical_price.get(ticker, {})
        sampled_volume = self._historical_volume.get(ticker, {})
        sampled_volume_net = self._historical_volume_net.get(ticker, {})

        # update sampled price, volume, net_volume
        baseline_timestamp = (timestamp // self.update_interval - 1) * self.update_interval
        sample_timestamp = timestamp // self.sample_interval * self.sample_interval
        sampled_price[sample_timestamp] = market_price
        sampled_volume[sample_timestamp] = sampled_volume.get(sample_timestamp, 0.) + volume
        sampled_volume_net[sample_timestamp] = sampled_volume.get(sample_timestamp, 0.) + volume * side

        for ts in list(sampled_price):
            if ts < baseline_timestamp:
                sampled_price.pop(ts)
                sampled_volume.pop(ts)
                sampled_volume_net.pop(ts)
            else:
                # since the sampled_price is always ordered, no need for further check
                break

        # update price pct change
        baseline_price = list(sampled_price.values())[0]  # the sampled_price must be ordered! Guaranteed by  consistency of the order of trade data
        price_change_pct = market_price / baseline_price - 1

        baseline_volume = sum(sampled_volume.values())
        net_volume = sum(sampled_volume_net.values())
        self._price_change_pct[ticker] = price_change_pct
        self._volume_pct[ticker] = net_volume / baseline_volume

        self._historical_price[ticker] = sampled_price
        self._historical_volume[ticker] = sampled_volume
        self._historical_volume_net[ticker] = sampled_volume_net

    @property
    def value(self) -> float:
        price_pct_change = []
        volume_pct = []

        if self.weights:
            for ticker in self.weights:
                if ticker in self._price_change_pct:
                    price_pct_change.append(self.weights[ticker] * self._price_change_pct[ticker])
                    volume_pct.append(self.weights[ticker] * self._volume_pct[ticker])
        else:
            price_pct_change.extend(self._price_change_pct.values())
            volume_pct.extend(self._volume_pct.values())

        slope = self.regression(x=volume_pct, y=price_pct_change)

        return slope


class MACDMonitor(MarketDataMonitor):
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

    def __init__(self, update_interval: float, name: str = 'Monitor.TA.MACD', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

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
    def is_ready(self) -> bool:
        return self.is_ready


class MACDIndexMonitor(MACDMonitor):
    """
    adjusted macd value (by its market_price), weighted average into a value of index
    """

    def __init__(self, update_interval: float, weights: dict[str, float], name: str = 'Monitor.TA.MACD.Index', monitor_id: str = None):
        super().__init__(
            update_interval=update_interval,
            name=name,
            monitor_id=monitor_id
        )

        self.weights = weights

    @property
    def value(self) -> float:
        index_value = 0.

        for ticker in self._macd:
            _ = self._macd[ticker]
            index_value += _.macd_histogram / _.close_price * self.weights.get(ticker, 0.)

        return index_value


class AggressivenessMonitor(MarketDataMonitor):
    """
    monitor the aggressiveness of buy / sell trades. the deeper the orders fill at one time, the more aggressive they are

    this module requires 2 additional fields
    - sell_order_id
    - buy_order_id

    without these 2 fields the monitor is set to is_ready = False
    the monitor aggregated aggressive buying / selling volume separately
    """

    def __init__(self, name: str = 'Monitor.Aggressiveness', monitor_id: str = None):
        super().__init__(
            name=name,
            monitor_id=monitor_id,
            mds=MDS
        )

        self._last_update: dict[str, float] = dict()
        self._trade_price: dict[str, dict[int, float]] = dict()
        self._aggressive_buy: dict[str, float] = {}
        self._aggressive_sell: dict[str, float] = {}
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        if isinstance(market_data, TradeData):
            trade_data = market_data
            ticker = trade_data.ticker
            price = trade_data.market_price
            volume = trade_data.volume
            side = trade_data.side.sign
            timestamp = trade_data.timestamp

            if ticker in self._trade_price:
                trade_price_log = self._trade_price[ticker]
            else:
                trade_price_log = self._trade_price[ticker] = {}

            if ticker not in self._last_update or self._last_update[ticker] < timestamp:
                trade_price_log.clear()
                return

            if side > 0:  # a buy init trade
                if 'buy_order_id' not in trade_data.additional:
                    self._is_ready = False

                order_id = trade_data.additional['buy_order_id']
            elif side < 0:
                if 'sell_order_id' not in trade_data.additional:
                    self._is_ready = False

                order_id = trade_data.additional['sell_order_id']
            else:
                return

            if order_id not in trade_price_log:
                trade_price_log[order_id] = price
            else:
                last_price = trade_price_log[order_id]

                if last_price == price:
                    pass
                else:
                    self._update_aggressiveness(ticker=ticker, side=side, volume=volume, timestamp=timestamp)

            self._last_update[ticker] = timestamp

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        if side > 0:
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker) + volume
        else:
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker) + volume

    @property
    def value(self) -> tuple[dict[str, float], dict[str, float]]:
        return self._aggressive_buy, self._aggressive_sell

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class AggressivenessEMAMonitor(AggressivenessMonitor):
    """
    ema average of aggressiveness buying / selling volume.

    Note that the discount process is triggered by on_trade_data.
    - Upon discount, there should be a discontinuity of index value.
    - But if the update interval is small enough, the effect should be marginal
    """

    def __init__(self, update_interval: float, alpha: float, name: str = 'Monitor.Aggressiveness.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

        self.update_interval = update_interval
        self.alpha = alpha
        self._last_discounted_update = dict()

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.name} should have an alpha from 0 to 1')

        if update_interval <= 0:
            LOGGER.warning(f'{self.name} should have a positive update_interval')

    def _discount_aggressiveness(self, ticker: str, timestamp: float):
        last_update = self._last_discounted_update.get(ticker, 0.)

        if last_update + self.update_interval < timestamp:
            time_span = timestamp - last_update
            adjust_power = time_span // self.update_interval
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) * (self.alpha ** adjust_power)
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) * (self.alpha ** adjust_power)
            self._last_discounted_update[ticker] = timestamp // self.update_interval * self.update_interval

    def _check_discontinuity(self, timestamp: float, tolerance: int = 1):
        discontinued = []

        for ticker in self._last_discounted_update:
            last_update = self._last_discounted_update[ticker]

            if last_update + tolerance * (self.update_interval + 1) < timestamp:
                discontinued.append(ticker)

        return discontinued

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        alpha = 1 if self._last_discounted_update.get(ticker, 0.) < timestamp else self.alpha

        if side > 0:
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker) + volume * alpha
        else:
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker) + volume * alpha


def register_monitor(index_weights: dict[str, float]) -> dict[str, MarketDataMonitor]:
    monitors = {}

    # trade flow monitor
    _ = TradeFlowMonitor()
    monitors[_.name] = _
    MDS.add_monitor(_)

    # trade flow ema monitor
    _ = TradeFlowEMAMonitor(update_interval=1, alpha=0.9885)  # alpha = 0.5 for each minute
    monitors[_.name] = _
    MDS.add_monitor(_)

    # price coherence monitor
    _ = PriceCoherenceMonitor(update_interval=60, sample_interval=1, weights=None)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # synthetic index monitor
    if index_weights:
        _ = SyntheticIndexMonitor(index_name='SZ50', weights=index_weights)
        monitors[_.name] = _
        MDS.add_monitor(_)

    # trade coherence monitor
    _ = TradeCoherenceMonitor(update_interval=60, sample_interval=1, weights=None)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # MACD monitor
    _ = MACDMonitor(update_interval=60)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # MACD index monitor
    if index_weights:
        _ = MACDIndexMonitor(weights=index_weights, update_interval=60)
        monitors[_.name] = _
        MDS.add_monitor(_)

    # aggressiveness monitor
    _ = AggressivenessMonitor()
    monitors[_.name] = _
    MDS.add_monitor(_)

    # aggressiveness ema monitor
    _ = AggressivenessEMAMonitor(update_interval=1, alpha=0.9885)  # alpha = 0.5 for each minute
    monitors[_.name] = _
    MDS.add_monitor(_)

    return monitors
