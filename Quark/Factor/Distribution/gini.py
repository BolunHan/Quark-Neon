from typing import Iterable

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData

from .. import MDS, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, Synthetic


class GiniMonitor(MarketDataMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, name: str = 'Monitor.Gini.PricePct', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price', mode='update')

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        market_price = market_data.market_price

        timestamp = market_data.timestamp
        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

    def clear(self) -> None:
        FixedIntervalSampler.clear(self)

    def gini_impurity(self, ticker: str = None, drop_last: bool = False) -> dict[str, float]:
        historical_price = self.get_sampler(name='price')
        gini_dict = {}

        if ticker is None:
            tasks = list(historical_price)
        elif isinstance(ticker, str):
            tasks = [ticker]
        elif isinstance(ticker, Iterable):
            tasks = list(ticker)
        else:
            raise TypeError(f'Invalid ticker {ticker}, expect str, list[str] or None.')

        for ticker in tasks:
            price_vector = list(historical_price[ticker].values())

            if drop_last:
                price_vector.pop(-1)

            if len(price_vector) < 3:
                continue

            price_pct_vector = np.diff(price_vector) / price_vector[:-1]

            n = len(price_pct_vector)
            n_up = sum([1 for _ in price_pct_vector if _ > 0])
            n_down = sum([1 for _ in price_pct_vector if _ < 0])

            gini = 1 - (n_up / n) ** 2 - (n_down / n) ** 2
            gini_dict[ticker] = gini

        return gini_dict

    @property
    def value(self) -> dict[str, float]:
        gini = self.gini_impurity()
        return gini

    @property
    def is_ready(self) -> bool:
        for _ in historical_price.values():
            if len(_) < 3:
                return False

        return self._is_ready


class GiniIndexMonitor(GiniMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float] = None, name: str = 'Monitor.Gini.PricePct.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        super().clear()
        Synthetic.clear(self)

    @property
    def value(self) -> float:
        gini_impurity = self.gini_impurity()
        return np.log2(self.composite(values=gini_impurity))


class GiniAdaptiveMonitor(GiniMonitor, AdaptiveVolumeIntervalSampler):
    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, name: str = 'Monitor.Gini.PricePct.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size, baseline_window=baseline_window, aligned_interval=aligned_interval)

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        super().clear()
        AdaptiveVolumeIntervalSampler.clear(self)

    @property
    def is_ready(self) -> bool:
        for ticker in self._volume_baseline['obs_vol_acc']:
            if ticker not in self._volume_baseline['sampling_interval']:
                return False

        return self._is_ready


class GiniIndexAdaptiveMonitor(GiniAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.Gini.PricePct.Index.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        super().clear()
        Synthetic.clear(self)

    @property
    def value(self) -> float:
        gini = self.gini_impurity()
        return self.composite(values=gini)
