import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData

from .. import EMA, MDS, LOGGER


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

    def clear(self):
        self._historical_price.clear()
        self._price_change_pct.clear()

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

    def clear(self):
        super().clear()
        EMA.clear(self)

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
