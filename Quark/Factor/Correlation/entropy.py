from __future__ import annotations

import json

import numpy as np
from PyQuantKit import MarketData

from .. import FactorMonitor, EMA, FixedIntervalSampler, AdaptiveVolumeIntervalSampler


class EntropyMonitor(FactorMonitor, FixedIntervalSampler):
    """
    Monitors and measures the entropy of the covariance matrix.

    The entropy measures the information coming from two parts:
    - The variance of the series.
    - The inter-connection of the series.

    If we ignore the primary trend, which is mostly the standard deviation (std),
    the entropy mainly denotes the coherence of the price vectors.

    A large entropy generally indicates the end of a trend.

    Attributes:
        sample_size (int): Max sample size.
        sampling_interval (float): Time interval for sampling market data.
        weights (dict[str, float]): Weights for individual stocks in the pool.
        ignore_primary (bool): Whether to ignore the primary component (std) of the covariance matrix.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
    """

    def __init__(self, sampling_interval: float, sample_size: int = 20, weights: dict[str, float] = None, pct_change: bool = False, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.weights = weights
        self.pct_change = pct_change
        self.ignore_primary = ignore_primary

        self.register_sampler(name='price', mode='update')

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
        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

    def clear(self) -> None:
        FixedIntervalSampler.clear(self)

        self.register_sampler(name='price', mode='update')

    @classmethod
    def covariance_matrix(cls, vectors: list[list[float]] | np.ndarray) -> np.ndarray:
        """
        Calculates the covariance matrix of the given vectors.

        Args:
            vectors (list[list[float]]): List of vectors.

        Returns:
            np.ndarray: Covariance matrix.
        """
        data = np.array(vectors)
        matrix = np.cov(data, ddof=0, rowvar=True)
        return matrix

    @classmethod
    def entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        """
        Calculates the entropy of the given covariance matrix.

        Args:
            matrix (list[list[float]] | np.ndarray): Covariance matrix.

        Returns:
            float: Entropy value.
        """
        # Note: The matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # Just to be safe
        e = e[e > 0]

        t = e * np.log2(e)
        return -np.sum(t)

    @classmethod
    def secondary_entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        """
        Calculates the secondary entropy of the given covariance matrix.

        Args:
            matrix (list[list[float]] | np.ndarray): Covariance matrix.

        Returns:
            float: Secondary entropy value.
        """
        # Note: The matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # Just to be safe
        e = e[e > 0]

        # Remove the primary component (std) of the covariance matrix
        primary_index = np.argmax(e)
        e = np.delete(e, primary_index)

        t = e * np.log2(e)
        return -np.sum(t)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}'
        ]

    def _param_static(self) -> dict[str, ...]:
        param_static = super()._param_static()

        param_static.update(
            weights=self.weights,
            pct_change=self.pct_change,
            ignore_primary=self.ignore_primary
        )

        return param_static

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            weights=dict(self.weights),
            pct_change=self.pct_change,
            ignore_primary=self.ignore_primary
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @property
    def value(self) -> float:
        """
        Calculates and returns the entropy value.

        Returns:
            float: Entropy value.
        """
        historical_price = self.get_sampler(name='price')
        price_matrix = []
        weight_vector = []

        if self.weights:
            vector_length = min([len(historical_price.get(ticker, {})) for ticker in self.weights])

            if vector_length < 3:
                return np.nan

            for ticker in self.weights:
                if ticker in historical_price:
                    price_matrix.append(list(historical_price[ticker])[-vector_length:])
                    weight_vector.append(self.weights[ticker])
        else:
            vector_length = min([len(_) for _ in historical_price.values()])

            if vector_length < 3:
                return np.nan

            price_matrix.extend([list(historical_price[ticker])[-vector_length:] for ticker in historical_price])
            weight_vector.extend([1] * len(historical_price))

        if len(price_matrix) < 3:
            return np.nan

        data_matrix = []
        for price_vector, weight in zip(price_matrix, weight_vector):
            price_vector = np.array(price_vector)
            if self.pct_change:
                data_matrix.append(np.diff(price_vector) / price_vector[:-1] * weight)
            else:
                data_matrix.append(price_vector * weight)

        cov = self.covariance_matrix(vectors=data_matrix)

        if self.ignore_primary:
            entropy = self.secondary_entropy(matrix=cov)
        else:
            entropy = self.entropy(matrix=cov)

        return entropy

    @property
    def is_ready(self) -> bool:
        """
        Checks if the EntropyMonitor is ready.

        Returns:
            bool: True if the monitor is ready, False otherwise.
        """
        for _ in self.get_sampler(name='price').values():
            if len(_) < 3:
                return False

        return True


class EntropyAdaptiveMonitor(EntropyMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 100, weights: dict[str, float] = None, pct_change: bool = True, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            weights=weights,
            pct_change=pct_change,
            ignore_primary=ignore_primary,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size, baseline_window=baseline_window)

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        AdaptiveVolumeIntervalSampler.clear(self)

        super().clear()

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class EntropyEMAMonitor(EntropyMonitor, EMA):
    """
    Monitors the Exponential Moving Average (EMA) of the entropy monitor.

    Inherits from EntropyMonitor and EMA classes.
    """

    def __init__(self, sampling_interval: float, sample_size: int, alpha: float, weights: dict[str, float] = None, pct_change: bool = True, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price.EMA', monitor_id: str = None):
        """
        Initializes the EntropyEMAMonitor.

        Args:
            sample_size (int): Max sample size.
            alpha (float): Exponential moving average smoothing factor.
            sampling_interval (float): Time interval for sampling market data.
            weights (dict): Weights for individual stocks in the pool.
            ignore_primary (bool): Whether to ignore the primary component (std) of the covariance matrix.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, weights=weights, pct_change=pct_change, ignore_primary=ignore_primary, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, alpha=alpha)

        self.entropy_ema = self.register_ema(name='entropy')

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'price':
            return

        entropy = super().value
        self.update_ema(ticker='entropy', entropy=entropy)

    def clear(self):
        """
        Clears historical price, price change, and EMA data.
        """
        EMA.clear(self)

        super().clear()

        self.entropy_ema = self.register_ema(name='entropy')

    @property
    def value(self) -> float:
        """
        Calculates and returns the EMA of entropy.

        Returns:
            float: EMA of entropy.
        """
        entropy = super().value
        return self.entropy_ema.get('entropy', np.nan)
