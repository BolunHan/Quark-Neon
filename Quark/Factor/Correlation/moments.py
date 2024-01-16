from collections import deque
from typing import Iterable

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData
from scipy.stats import skew

from .. import MDS, FixTemporalIntervalMonitor, AdaptiveVolumeIntervalMonitor, Synthetic


class SkewnessMonitor(MarketDataMonitor, FixTemporalIntervalMonitor, Synthetic):

    def __init__(self, update_interval: float, sample_interval: float = 1., weights: dict[str, float] = None, name: str = 'Monitor.Skewness.PricePct', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        FixTemporalIntervalMonitor.__init__(self=self, update_interval=update_interval, sample_interval=sample_interval)
        Synthetic.__init__(self=self, weights=weights)

        self._historical_price = {}
        self._historical_skewness: dict[str, deque[float]] = {}

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

    def on_entry_add(self, ticker: str, key, value):
        super().on_entry_add(ticker=ticker, key=key, value=value)
        skewness_dict = self.skewness(ticker=ticker)

        if ticker not in skewness_dict:
            return

        skewness = skewness_dict[ticker]

        if not np.isfinite(skewness):
            return

        if ticker in self._historical_skewness:
            historical_skewness = self._historical_skewness[ticker]
        else:
            historical_skewness = self._historical_skewness[ticker] = deque(maxlen=int(self.update_interval // self.sample_interval))

        historical_skewness.append(skewness)

    def slope(self) -> dict[str, float]:
        slope_dict = {}
        for ticker in self._historical_skewness:
            skewness = list(self._historical_skewness[ticker])

            if len(skewness) < 3:
                slope = np.nan
            else:
                x = list(range(len(skewness)))
                x = np.vstack([x, np.ones(len(x))]).T
                y = np.array(skewness)

                slope, c = np.linalg.lstsq(x, y, rcond=None)[0]

            slope_dict[ticker] = slope

        return slope_dict

    def skewness(self, ticker: str = None) -> dict[str, float]:
        skewness_dict = {}

        if ticker is None:
            tasks = list(self._historical_skewness)
        elif isinstance(ticker, str):
            tasks = [ticker]
        elif isinstance(ticker, Iterable):
            tasks = list(ticker)
        else:
            raise TypeError(f'Invalid ticker {ticker}, expect str, list[str] or None.')

        for ticker in tasks:
            price_vector = list(self._historical_price[ticker].values())

            if len(price_vector) < 3:
                continue

            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            skewness: float = skew(price_pct_vector, bias=True, nan_policy='omit')
            skewness_dict[ticker] = skewness

        return skewness_dict

    @property
    def value(self) -> dict[str, float]:
        skewness = self.skewness()
        skewness.update({'Index': self.composite(values=skewness), 'Slope': self.composite(values=self.slope())})
        return skewness

    @property
    def is_ready(self) -> bool:
        for _ in self._historical_price.values():
            if len(_) < 3:
                return False

        return self._is_ready


class SkewnessAdaptiveMonitor(SkewnessMonitor, AdaptiveVolumeIntervalMonitor):
    def __init__(self, update_interval: float, sample_rate: float = 20., baseline_window: int = 5, weights: dict[str, float] = None, name: str = 'Monitor.Skewness.PricePct.Adaptive', monitor_id: str = None):
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
