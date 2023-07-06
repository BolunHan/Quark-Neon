"""
implement MarketDataMonitor and add it to the register() method
"""
import abc
import datetime
from collections import deque

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData, BarData

from . import LOGGER, MDS
from .decoder import OnlineDecoder, Wavelet
from ..Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE


class IndexWeight(dict):
    def __init__(self, index_name: str, *args, **kwargs):
        self.index_name = index_name

        super().__init__(*args, **kwargs)

    def normalize(self):
        total_weight = sum(list(self.values()))

        if not total_weight:
            return

        for _ in self:
            self[_] /= total_weight

    @property
    def components(self) -> list[str]:
        return list(self.keys())


class EMA(metaclass=abc.ABCMeta):
    def __init__(self, discount_interval: float, alpha: float = None, window: int = None):
        self.discount_interval = discount_interval
        self.alpha = alpha if alpha else 1 - 2 / (window + 1)
        self.window = window if window else round(2 / (1 - alpha) - 1)

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.__class__.__name__} should have an alpha from 0 to 1')

        if discount_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive discount_interval')

        self._last_discount_ts: dict[str, float] = {}
        self._history: dict[str, dict[str, float]] = {}
        self._current: dict[str, dict[str, float]] = {}
        self._window: dict[str, dict[str, deque[float]]] = {}
        self.ema: dict[str, dict[str, float]] = {}

    def _register_ema(self, name):
        self._history[name] = {}
        self._current[name] = {}
        self._window[name] = {}
        _ = self.ema[name] = {}
        return _

    def _update_ema(self, ticker: str, timestamp: float = None, **update_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {update_data}')
                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)

        # update to current
        for entry_name in update_data:
            if entry_name in self._current:
                if np.isfinite(_ := update_data[entry_name]):
                    current = self._current[entry_name][ticker] = _
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = np.nan
                    else:
                        self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

    def _accumulate_ema(self, ticker: str, timestamp: float = None, **accumulative_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {accumulative_data}')

                time_span = max(0, last_discount - timestamp)
                adjust_factor = time_span // self.discount_interval
                alpha = self.alpha ** adjust_factor

                for entry_name in accumulative_data:
                    if entry_name in self._history:
                        if np.isfinite(_ := accumulative_data[entry_name]):
                            current = self._current[entry_name].get(ticker, 0.)
                            memory = self._history[entry_name][ticker] = self._history[entry_name].get(ticker, 0.) + _ * alpha

                            if memory is None:
                                self.ema[entry_name][ticker] = np.nan
                            else:
                                self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)
        # add to current
        for entry_name in accumulative_data:
            if entry_name in self._current:
                if np.isfinite(_ := accumulative_data[entry_name]):
                    current = self._current[entry_name][ticker] = self._current[entry_name].get(ticker, 0.) + _
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = np.nan
                    else:
                        self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

    def _discount_ema(self, ticker: str, timestamp: float):
        last_update = self._last_discount_ts.get(ticker, timestamp)

        # a discount event is triggered
        if last_update + self.discount_interval <= timestamp:
            time_span = timestamp - last_update
            adjust_power = int(time_span // self.discount_interval)

            for entry_name in self._history:
                current = self._current[entry_name].get(ticker)
                memory = self._history[entry_name].get(ticker)
                window: deque = self._window[entry_name].get(ticker, deque(maxlen=self.window))

                # pre-check: drop None or nan
                if current is None or not np.isfinite(current):
                    return

                # step 1: update window
                for _ in range(adjust_power - 1):
                    if window:
                        window.append(window[-1])

                window.append(current)

                # step 2: re-calculate memory if window is not full
                if len(window) < window.maxlen or memory is None:
                    memory = None

                    for _ in window:
                        if memory is None:
                            memory = _

                        memory = memory * self.alpha + _ * (1 - self.alpha)

                # step 3: calculate ema value by memory and current value
                ema = memory * self.alpha + current * (1 - self.alpha)

                # step 4: update EMA state
                self._current[entry_name].pop(ticker)
                self._history[entry_name][ticker] = ema
                self._window[entry_name][ticker] = window
                self.ema[entry_name][ticker] = ema

        self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

    def _check_discontinuity(self, timestamp: float, tolerance: int = 1):
        discontinued = []

        for ticker in self._last_discount_ts:
            last_update = self._last_discount_ts[ticker]

            if last_update + (tolerance + 1) * self.discount_interval < timestamp:
                discontinued.append(ticker)

        return discontinued

    def _discount_all(self, timestamp: float):
        for _ in self._check_discontinuity(timestamp=timestamp, tolerance=1):
            self._discount_ema(ticker=_, timestamp=timestamp)


class Synthetic(metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price = 1.

    def _update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        if ticker not in self.base_price:
            self.base_price[ticker] = market_price

        self.last_price[ticker] = market_price

    @property
    def synthetic_price(self):
        price_list = []
        weight_list = []

        for ticker in self.weights:
            weight_list.append(self.weights[ticker])

            if ticker in self.last_price:
                price_list.append(self.last_price[ticker] / self.base_price[ticker])
            else:
                price_list.append(1.)

        synthetic_price = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        return synthetic_price


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


class TradeFlowEMAMonitor(TradeFlowMonitor, EMA):
    """
    ema discounted trade flow
    Note that the discount process is triggered by on_trade_data. Upon discount, there should be a discontinuity of index value.
    But if the update interval is small enough, the effect should be marginal

    trade flow is a relative stable indication. a large positive aggressiveness indicates market is in an upward ongoing trend
    """

    def __init__(self, discount_interval: float, alpha: float, normalized: bool = True, name: str = 'Monitor.TradeFlow.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

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


class CoherenceMonitor(MarketDataMonitor):
    """
    measure the coherence of price pct change
    the basic assumption is that for a stock pool, rank the gaining / losing stock by this price_change_pct

    the rank and log(price_pct) should have a linear correlation.

    the correlation coefficient should give a proximate indication of price change coherence

    a factor of up_dispersion / (up_dispersion + down_dispersion) is an indication of (start of) a downward trend
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

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

        self._log_price(ticker=ticker, market_price=market_price, timestamp=timestamp)

        # update price pct change
        baseline_price = list(self._historical_price[ticker].values())[0]  # the sampled_price must be ordered! Guaranteed by  consistency of the order of trade data
        price_change_pct = market_price / baseline_price - 1
        self._price_change_pct[ticker] = price_change_pct

    def _log_price(self, ticker: str, market_price: float, timestamp: float):

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

        self._historical_price[ticker] = sampled_price

    @classmethod
    def regression(cls, y: list[float] | np.ndarray, x: list[float] | np.ndarray = None):
        y = np.array(y)
        x = np.array(x)

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator:
            slope = numerator / denominator
        else:
            slope = np.nan

        return slope

    def collect_dispersion(self, side: int):
        price_change_list = []
        weights = []
        if self.weights:
            for ticker in self.weights:

                if ticker not in self._price_change_pct:
                    continue

                price_change = self._price_change_pct[ticker]

                if price_change * side <= 0:
                    continue

                price_change_list.append(price_change)
                weights.append(self.weights[ticker])

            weights = np.sqrt(weights)

            y = np.array(price_change_list) * weights
            x = (np.argsort(y).argsort() + 1) * weights
        else:
            y = np.array(price_change_list)
            x = np.argsort(y).argsort() + 1

        if len(x) < 3:
            return np.nan

        return self.regression(y=y, x=x)

    @property
    def value(self) -> tuple[float, float]:
        up_dispersion = self.collect_dispersion(side=1)
        down_dispersion = self.collect_dispersion(side=-1)

        return up_dispersion, down_dispersion

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class CoherenceEMAMonitor(CoherenceMonitor, EMA):
    """
    the ema of coherence monitor.
    """

    def __init__(self, update_interval: float, discount_interval: float, alpha: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price.EMA', monitor_id: str = None):
        super().__init__(update_interval=update_interval, sample_interval=sample_interval, weights=weights, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

        self.dispersion_ratio = self._register_ema(name='dispersion_ratio')
        self.last_update = 0.

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker='dispersion_ratio', timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

        if self.last_update + self.update_interval < timestamp:
            _ = self.value
            self.last_update = timestamp // self.update_interval * self.update_interval

    @property
    def value(self) -> tuple[float, float, float]:
        up_dispersion = self.collect_dispersion(side=1)
        down_dispersion = self.collect_dispersion(side=-1)

        if up_dispersion < 0:
            dispersion_ratio = 1.
        elif down_dispersion < 0:
            dispersion_ratio = 0.
        else:
            dispersion_ratio = down_dispersion / (up_dispersion + down_dispersion)

        self._update_ema(ticker='dispersion_ratio', dispersion_ratio=dispersion_ratio - 0.5)

        return up_dispersion, down_dispersion, self.dispersion_ratio.get('dispersion_ratio', np.nan)


class TradeCoherenceMonitor(CoherenceMonitor):
    """
    similar like the price coherence, the price_change_pct should also have a linear correlation of net trade flow pct

    a large, negative trade coherence generally implies a strong, ongoing trend
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

        super().__call__(market_data=market_data)

        if isinstance(market_data, TradeData):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        sampled_volume = self._historical_volume.get(ticker, {})
        sampled_volume_net = self._historical_volume_net.get(ticker, {})

        # update sampled price, volume, net_volume
        baseline_timestamp = (timestamp // self.update_interval - 1) * self.update_interval
        sample_timestamp = timestamp // self.sample_interval * self.sample_interval
        sampled_volume[sample_timestamp] = sampled_volume.get(sample_timestamp, 0.) + volume
        sampled_volume_net[sample_timestamp] = sampled_volume.get(sample_timestamp, 0.) + volume * side

        for ts in list(sampled_volume):
            if ts < baseline_timestamp:
                sampled_volume.pop(ts)
                sampled_volume_net.pop(ts)
            else:
                # since the sampled_price is always ordered, no need for further check
                break

        # update price pct change

        baseline_volume = sum(sampled_volume.values())
        net_volume = sum(sampled_volume_net.values())
        self._volume_pct[ticker] = net_volume / baseline_volume

        self._historical_volume[ticker] = sampled_volume
        self._historical_volume_net[ticker] = sampled_volume_net

    @property
    def value(self) -> float:
        price_pct_change = []
        volume_pct = []

        if self.weights:
            for ticker in self.weights:
                if ticker in self._price_change_pct:
                    # noted, the weight for each observation is its frequency, which denotes a scale of sqrt(weights)
                    price_pct_change.append(np.sqrt(self.weights[ticker]) * self._price_change_pct[ticker])
                    volume_pct.append(np.sqrt(self.weights[ticker]) * self._volume_pct[ticker])
        else:
            price_pct_change.extend(self._price_change_pct.values())
            volume_pct.extend(self._volume_pct.values())

        slope = self.regression(x=volume_pct, y=price_pct_change)

        return slope


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

        index_price = self.synthetic_price

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
        return self.synthetic_price

    @property
    def active_bar(self):
        return self._active_bar_data


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

    def __init__(self, update_interval: float, weights: dict[str, float] = None, name: str = 'Monitor.TA.MACD', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

        self.update_interval = update_interval
        self.weights = weights

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
    def weighted_index(self) -> float:
        macd_value = self.value
        weighted_index = 0.

        if self.weights is not None:
            for ticker in self.weights:
                weighted_index += macd_value.get(ticker, 0.) * self.weights[ticker]
        else:
            weighted_index = np.nanmean(macd_value.values())

        return weighted_index

    @property
    def is_ready(self) -> bool:
        return self._is_ready


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
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        price = trade_data.market_price
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        if ticker in self._trade_price:
            trade_price_log = self._trade_price[ticker]
        else:
            trade_price_log = self._trade_price[ticker] = {}

        # trade logs triggered by the same order can not have 2 different timestamp
        if ticker not in self._last_update or self._last_update[ticker] < timestamp:
            trade_price_log.clear()
            self._last_update[ticker] = timestamp
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
                self._update_aggressiveness(
                    ticker=ticker,
                    side=side,
                    volume=volume,
                    timestamp=timestamp
                )

        self._last_update[ticker] = timestamp

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        if side > 0:
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) + volume
        else:
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) + volume

    @property
    def value(self) -> tuple[dict[str, float], dict[str, float]]:
        return self._aggressive_buy, self._aggressive_sell

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class AggressivenessEMAMonitor(AggressivenessMonitor, EMA):
    """
    ema average of aggressiveness buying / selling volume.

    Note that the discount process is triggered by on_trade_data.
    - Upon discount, there should be a discontinuity of index value.
    - But if the update interval is small enough, the effect should be marginal

    aggressiveness is a relative stable indication. a large positive aggressiveness indicates market is in an upward ongoing trend
    """

    def __init__(self, discount_interval: float, alpha: float, normalized: bool = True, name: str = 'Monitor.Aggressiveness.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

        self.normalized = normalized

        self._aggressive_buy: dict[str, float] = self._register_ema(name='aggressive_buy')
        self._aggressive_sell: dict[str, float] = self._register_ema(name='aggressive_sell')
        self._trade_volume = self._register_ema(name='trade_volume')

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        total_volume = self._trade_volume[ticker]
        adjusted_volume = volume / total_volume if total_volume else 0.

        if side > 0:
            self._accumulate_ema(ticker=ticker, timestamp=timestamp, aggressive_buy=adjusted_volume, aggressive_sell=0.)
        else:
            self._accumulate_ema(ticker=ticker, timestamp=timestamp, aggressive_buy=0., aggressive_sell=adjusted_volume)

    def _on_trade(self, trade_data: TradeData):
        ticker = trade_data.ticker
        self._accumulate_ema(ticker=ticker, trade_volume=trade_data.volume)  # to avoid redundant calculation, timestamp is not passed-in, so that the discount function will not be triggered
        super()._on_trade(trade_data=trade_data)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker=ticker, timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        return super().__call__(market_data=market_data, **kwargs)


class EntropyMonitor(CoherenceMonitor):
    """
    measure the entropy of covariance matrix

    the entropy measure the information coming from 2 part:
    - the variance of the series
    - the inter-connection of the series

    if we ignore the primary trend, which is mostly the std, the entropy mainly denoted the coherence of the price vectors

    a large entropy generally indicate the end of a trend
    """

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, normalized: bool = True, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price', monitor_id: str = None):

        super().__init__(
            update_interval=update_interval,
            sample_interval=sample_interval,
            weights=weights,
            name=name,
            monitor_id=monitor_id
        )

        self.normalized = normalized
        self.ignore_primary = ignore_primary

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        if ticker not in self.weights:
            return

        market_price = market_data.market_price * self.weights[ticker]
        timestamp = market_data.timestamp

        self._log_price(ticker=ticker, market_price=market_price, timestamp=timestamp)

    @classmethod
    def covariance_matrix(cls, vectors: list[list[float]]):
        data = np.array(vectors)
        matrix = np.cov(data, ddof=0, rowvar=True)
        return matrix

    @classmethod
    def entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        # noted, the matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # just to be safe
        e = e[e > 0]

        t = e * np.log(e)
        return -np.sum(t)

    @classmethod
    def secondary_entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        # noted, the matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # just to be safe
        e = e[e > 0]

        # Remove the primary component (std) of the covariance matrix
        primary_index = np.argmax(e)
        e = np.delete(e, primary_index)

        t = e * np.log(e)
        return -np.sum(t)

    @property
    def value(self) -> float:
        price_vector = []

        if self.weights:
            vector_length = min([len(self._historical_price.get(_, {})) for _ in self.weights])

            if vector_length < 3:
                return np.nan

            for ticker in self.weights:
                if ticker in self._historical_price:
                    price_vector.append(list(self._historical_price[ticker].values())[-vector_length:])
        else:
            vector_length = min([len(_) for _ in self._historical_price.values()])

            if vector_length < 3:
                return np.nan

            price_vector.extend([list(self._historical_price[ticker].values())[-vector_length:] for ticker in self._historical_price])

        if len(price_vector) < 3:
            return np.nan

        cov = self.covariance_matrix(vectors=price_vector)

        if self.ignore_primary:
            entropy = self.secondary_entropy(matrix=cov)
        else:
            entropy = self.entropy(matrix=cov)

        return entropy

    @property
    def is_ready(self) -> bool:
        for _ in self._historical_price.values():
            if len(_) < 3:
                return False

        return super().is_ready


class EntropyEMAMonitor(EntropyMonitor, EMA):
    """
    the ema of entropy monitor
    """

    def __init__(self, update_interval: float, discount_interval: float, alpha: float, sample_interval: float = 1., weights: dict[str, float] = None, normalized: bool = True, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price.EMA', monitor_id: str = None):
        super().__init__(update_interval=update_interval, sample_interval=sample_interval, weights=weights, normalized=normalized, ignore_primary=ignore_primary, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

        self.entropy_ema = self._register_ema(name='entropy')
        self.last_update = 0.

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker='entropy', timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

        if self.last_update + self.update_interval < timestamp:
            _ = self.value
            self.last_update = timestamp // self.update_interval * self.update_interval

    @property
    def value(self) -> float:
        entropy = super().value
        self._update_ema(ticker='entropy', entropy=entropy)
        return self.entropy_ema.get('entropy', np.nan)


class VolatilityMonitor(MarketDataMonitor, Synthetic):

    def __init__(self, weights: dict[str, float], name: str = 'Monitor.Volatility.Daily', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        Synthetic.__init__(self=self, weights=weights)

        self.daily_volatility: dict[str, float] = {}  # must be assigned from outside
        self.index_volatility: float = np.nan  # must be assigned from outside

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        self._update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)

    @property
    def value(self) -> dict[str, float]:
        volatility_adjusted = {}

        for ticker in self.weights:
            if ticker not in self.daily_volatility:
                continue

            if ticker not in self.last_price:
                continue

            volatility_adjusted[ticker] = (self.last_price[ticker] / self.base_price[ticker] - 1) / self.daily_volatility[ticker]

        return volatility_adjusted

    @property
    def weighted_index(self) -> float:
        volatility_adjusted = self.value
        weighted_index = 0.

        weighted_volatility = np.sum([self.weights[_] * self.daily_volatility.get(_, 0.) for _ in self.weights])
        diff_base = weighted_volatility - self.index_volatility

        for ticker in self.weights:
            weighted_index += volatility_adjusted.get(ticker, 0.) * self.weights[ticker]

        index_volatility_range = (self.synthetic_price / self.synthetic_base_price - 1) / weighted_volatility

        if not index_volatility_range:
            return 0.

        weighted_index -= index_volatility_range
        weighted_index -= diff_base

        return weighted_index

    @property
    def is_ready(self) -> bool:
        return self._is_ready


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
        self.update_decoder(ticker='synthetic', market_price=self.synthetic_price, timestamp=market_data.timestamp)

    @property
    def value(self) -> list[Wavelet]:
        return self.state_history['synthetic']


def register_monitor(index_name: str, index_weights: dict[str, float] = None) -> dict[str, MarketDataMonitor]:
    monitors = {}
    alpha_0 = 0.9885  # alpha = 0.5 for each minute
    alpha_1 = 0.9735  # alpha = 0.2 for each minute
    alpha_2 = 0.9624  # alpha = 0.1 for each minute
    alpha_3 = 0.9261  # alpha = 0.01 for each minute
    alpha_4 = 0.8913  # alpha = 0.001 for each minute

    index_weights = IndexWeight(index_name=index_name, **index_weights)
    index_weights.normalize()
    LOGGER.info(f'Register monitors for index {index_name} and its {len(index_weights.components)} components!')

    # trade flow monitor
    # _ = TradeFlowMonitor()
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # trade flow ema monitor
    _ = TradeFlowEMAMonitor(discount_interval=1, alpha=alpha_0)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # price coherence monitor
    # _ = CoherenceMonitor(update_interval=60, sample_interval=1, weights=index_weights)
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # price coherence ema monitor
    _ = CoherenceEMAMonitor(update_interval=60, sample_interval=1, weights=index_weights, discount_interval=1, alpha=alpha_4)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # synthetic index monitor
    _ = SyntheticIndexMonitor(index_name=index_name, weights=index_weights)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # trade coherence monitor
    _ = TradeCoherenceMonitor(update_interval=60, sample_interval=1, weights=index_weights)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # MACD monitor
    _ = MACDMonitor(weights=index_weights, update_interval=60)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # aggressiveness monitor
    # _ = AggressivenessMonitor()
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # aggressiveness ema monitor
    _ = AggressivenessEMAMonitor(discount_interval=1, alpha=alpha_4)  # alpha = 0.5 for each minute
    monitors[_.name] = _
    MDS.add_monitor(_)

    # price coherence monitor
    # _ = EntropyMonitor(update_interval=60, sample_interval=1, weights=index_weights)
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # price coherence monitor
    _ = EntropyEMAMonitor(update_interval=60, sample_interval=1, weights=index_weights, discount_interval=1, alpha=alpha_4)
    monitors[_.name] = _
    MDS.add_monitor(_)

    # price coherence monitor
    # _ = VolatilityMonitor(weights=index_weights)
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # price movement online decoder
    # _ = DecoderMonitor(retrospective=False)
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    # price movement online decoder
    # _ = IndexDecoderMonitor(up_threshold=0.005, down_threshold=0.005, confirmation_level=0.002, retrospective=True, weights=index_weights)
    # monitors[_.name] = _
    # MDS.add_monitor(_)

    return monitors
