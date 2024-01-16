from collections import deque

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData
from scipy.stats import skew

from .. import MDS, FixTemporalIntervalMonitor, AdaptiveVolumeIntervalMonitor


class SkewnessMonitor(MarketDataMonitor, FixTemporalIntervalMonitor):

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Skewness.Price', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        FixTemporalIntervalMonitor.__init__(self=self, update_interval=update_interval, sample_interval=sample_interval)

        self.weights = weights

        self._historical_price = {}
        self._historical_skewness = deque(maxlen=int(update_interval // sample_interval))

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the EntropyMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        market_price = market_data.market_price

        timestamp = market_data.timestamp
        self.log_obs(ticker=ticker, value=market_price, timestamp=timestamp, storage=self._historical_price)

    def clear(self) -> None:
        self._historical_price.clear()
        self._historical_skewness.clear()
        FixTemporalIntervalMonitor.clear(self)

    def on_entry_add(self, key, value):
        super().on_entry_add(key=key, value=value)
        skewness = self.skewness()
        self._historical_skewness.append(skewness)

    def slope(self) -> float:
        skewness = list(self._historical_skewness)

        if len(skewness) < 3:
            return np.nan

        x = list(range(len(skewness)))
        x = np.vstack([x, np.ones(len(x))]).T
        y = np.array(skewness)

        slope, c = np.linalg.lstsq(x, y, rcond=None)[0]
        return slope

    def skewness(self):
        price_matrix = []
        weight_vector = []

        if self.weights:
            vector_length = min([len(self._historical_price.get(ticker, {})) for ticker in self.weights])

            if vector_length < 3:
                return np.nan

            for ticker in self.weights:
                if ticker in self._historical_price:
                    price_matrix.append(list(self._historical_price[ticker].values())[-vector_length:])
                    weight_vector.append(self.weights[ticker])
        else:
            vector_length = min([len(_) for _ in self._historical_price.values()])

            if vector_length < 3:
                return np.nan

            price_matrix.extend([list(self._historical_price[ticker].values())[-vector_length:] for ticker in self._historical_price])
            weight_vector.extend([1] * len(self._historical_price))

        if len(price_matrix) < 3:
            return np.nan

        weighted_skewness = 0.
        for price_vector, weight in zip(price_matrix, weight_vector):
            price_vector = np.array(price_vector)
            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            skewness = skew(price_pct_vector, bias=True, nan_policy='omit')

            if np.isfinite(skewness):
                weighted_skewness += skewness * weight

        return weighted_skewness

    @property
    def value(self) -> dict[str, float]:
        return {'value': self.skewness(), 'slope': self.slope()}

    @property
    def is_ready(self) -> bool:
        for _ in self._historical_price.values():
            if len(_) < 3:
                return False

        return self._is_ready


class SkewnessAdaptiveMonitor(SkewnessMonitor, AdaptiveVolumeIntervalMonitor):
    def __init__(self, update_interval: float, sample_rate: float = 20., baseline_window: int = 5, weights: dict[str, float] = None, name: str = 'Monitor.Skewness.Price.Adaptive', monitor_id: str = None):
        super().__init__(
            update_interval=update_interval,
            sample_interval=update_interval / sample_rate,
            weights=weights,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalMonitor.__init__(self=self, update_interval=update_interval, sample_rate=sample_rate, baseline_window=baseline_window)

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        super().clear()
        AdaptiveVolumeIntervalMonitor.clear(self)

    @property
    def is_ready(self) -> bool:
        for ticker in self._volume_baseline['obs_acc_vol']:
            if ticker not in self._volume_baseline['baseline']:
                return False

        return self._is_ready
