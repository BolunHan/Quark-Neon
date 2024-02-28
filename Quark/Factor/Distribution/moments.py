from __future__ import annotations

import json
from collections import deque
from typing import Iterable, Self

import numpy as np
from PyQuantKit import MarketData
from scipy.stats import skew

from .. import FixedIntervalSampler, AdaptiveVolumeIntervalSampler, Synthetic, FactorMonitor


class SkewnessMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, name: str = 'Monitor.Skewness.PricePct', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price', mode='update')
        self._historical_skewness: dict[str, deque[float]] = {}

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        market_price = market_data.market_price

        timestamp = market_data.timestamp
        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            historical_skewness=self._historical_skewness
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def update_from_json(self, json_dict: dict) -> Self:
        super().update_from_json(json_dict=json_dict)
        self._historical_skewness.update(json_dict['historical_skewness'])
        return self

    def clear(self) -> None:
        FixedIntervalSampler.clear(self)

        self._historical_skewness.clear()

        self.register_sampler(name='price', mode='update')

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'price':
            return

        skewness_dict = self.skewness(ticker=ticker, drop_last=True)  # calculate the skewness without the new entry

        if ticker not in skewness_dict:
            return

        skewness = skewness_dict[ticker]

        if not np.isfinite(skewness):
            return

        if ticker in self._historical_skewness:
            historical_skewness = self._historical_skewness[ticker]
        else:
            historical_skewness = self._historical_skewness[ticker] = deque(maxlen=self.sample_size)

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

    def skewness(self, ticker: str = None, drop_last: bool = False) -> dict[str, float]:
        historical_price = self.get_sampler(name='price')
        skewness_dict = {}

        if ticker is None:
            tasks = list(historical_price)
        elif isinstance(ticker, str):
            tasks = [ticker]
        elif isinstance(ticker, Iterable):
            tasks = list(ticker)
        else:
            raise TypeError(f'Invalid ticker {ticker}, expect str, list[str] or None.')

        for ticker in tasks:
            price_vector = list(historical_price[ticker])

            if drop_last:
                price_vector.pop(-1)

            if len(price_vector) < 3:
                continue

            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            # noinspection PyTypeChecker
            skewness: float = skew(price_pct_vector, bias=True, nan_policy='omit')
            skewness_dict[ticker] = skewness

        return skewness_dict

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}'
        ]

    @property
    def value(self) -> dict[str, float]:
        skewness = self.skewness()
        return skewness

    @property
    def is_ready(self) -> bool:
        for _ in self.get_sampler(name='price').values():
            if len(_) < 3:
                return False

        return True


class SkewnessIndexMonitor(SkewnessMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float] = None, name: str = 'Monitor.Skewness.PricePct.Index', monitor_id: str = None):
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
        Synthetic.clear(self)

        super().clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index',
            f'{self.name.removeprefix("Monitor.")}.Slope'
        ]

    @property
    def value(self) -> dict[str, float]:
        skewness = self.skewness()
        return {'Index': self.composite(values=skewness), 'Slope': self.composite(values=self.slope())}


class SkewnessAdaptiveMonitor(SkewnessMonitor, AdaptiveVolumeIntervalSampler):
    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, name: str = 'Monitor.Skewness.PricePct.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval
        )

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        AdaptiveVolumeIntervalSampler.clear(self)

        super().clear()

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class SkewnessIndexAdaptiveMonitor(SkewnessAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.Skewness.PricePct.Index.Adaptive', monitor_id: str = None):
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
        Synthetic.clear(self)

        super().clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index',
            f'{self.name.removeprefix("Monitor.")}.Slope'
        ]

    @property
    def value(self) -> dict[str, float]:
        skewness = self.skewness()
        return {'Index': self.composite(values=skewness), 'Slope': self.composite(values=self.slope())}
