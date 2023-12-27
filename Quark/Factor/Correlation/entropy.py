"""
This script defines classes for monitoring and measuring the entropy of the covariance matrix
for a stock pool. It extends the CoherenceMonitor class and includes functionalities for calculating
covariance matrices, entropy, and exponential moving averages (EMA).

Classes:
- EntropyMonitor: Monitors and measures the entropy of the covariance matrix for price vectors.
- EntropyEMAMonitor: Extends EntropyMonitor and includes an EMA for entropy.

Usage:
1. Instantiate the desired monitor class with appropriate parameters.
2. Call the instance with market data to update the monitor.
3. Retrieve the entropy values using the 'value' property of the monitor instance.

Note: This script assumes the availability of PyQuantKit, numpy, CoherenceMonitor, and EMA.

Author: Bolun
Date: 2023-12-26
"""

import numpy as np
from PyQuantKit import MarketData

from .coherence import CoherenceMonitor
from .. import EMA


class EntropyMonitor(CoherenceMonitor):
    """
    Monitors and measures the entropy of the covariance matrix.

    The entropy measures the information coming from two parts:
    - The variance of the series.
    - The inter-connection of the series.

    If we ignore the primary trend, which is mostly the standard deviation (std),
    the entropy mainly denotes the coherence of the price vectors.

    A large entropy generally indicates the end of a trend.

    Attributes:
        update_interval (float): Time interval for updating the monitor.
        sample_interval (float): Time interval for sampling market data.
        weights (dict[str, float]): Weights for individual stocks in the pool.
        normalized (bool): Whether to normalize the covariance matrix.
        ignore_primary (bool): Whether to ignore the primary component (std) of the covariance matrix.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
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
        """
        Updates the EntropyMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker

        if ticker not in self.weights:
            return

        market_price = market_data.market_price * self.weights[ticker]
        timestamp = market_data.timestamp

        self._log_price(ticker=ticker, market_price=market_price, timestamp=timestamp)

    @classmethod
    def covariance_matrix(cls, vectors: list[list[float]]) -> np.ndarray:
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

        t = e * np.log(e)
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

        t = e * np.log(e)
        return -np.sum(t)

    @property
    def value(self) -> float:
        """
        Calculates and returns the entropy value.

        Returns:
            float: Entropy value.
        """
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
        """
        Checks if the EntropyMonitor is ready.

        Returns:
            bool: True if the monitor is ready, False otherwise.
        """
        for _ in self._historical_price.values():
            if len(_) < 3:
                return False

        return super().is_ready


class EntropyEMAMonitor(EntropyMonitor, EMA):
    """
    Monitors the Exponential Moving Average (EMA) of the entropy monitor.

    Inherits from EntropyMonitor and EMA classes.
    """

    def __init__(self, update_interval: float, discount_interval: float, alpha: float, sample_interval: float = 1., weights: dict[str, float] = None, normalized: bool = True, ignore_primary: bool = True, name: str = 'Monitor.Entropy.Price.EMA', monitor_id: str = None):
        """
        Initializes the EntropyEMAMonitor.

        Args:
            update_interval (float): Time interval for updating the monitor.
            discount_interval (float): Time interval for discounting EMA values.
            alpha (float): Exponential moving average smoothing factor.
            sample_interval (float): Time interval for sampling market data.
            weights (dict): Weights for individual stocks in the pool.
            normalized (bool): Whether to normalize the covariance matrix.
            ignore_primary (bool): Whether to ignore the primary component (std) of the covariance matrix.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(update_interval=update_interval, sample_interval=sample_interval, weights=weights, normalized=normalized, ignore_primary=ignore_primary, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)

        self.entropy_ema = self._register_ema(name='entropy')
        self.last_update = 0.

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Updates the EntropyEMAMonitor with market data.

        Args:
            market_data (MarketData): Market data object containing price information.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker='entropy', timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

        if self.last_update + self.update_interval < timestamp:
            _ = self.value
            self.last_update = timestamp // self.update_interval * self.update_interval

    def clear(self):
        """
        Clears historical price, price change, and EMA data.
        """
        super().clear()
        EMA.clear(self)

        self.entropy_ema = self._register_ema(name='entropy')
        self.last_update = 0.

    @property
    def value(self) -> float:
        """
        Calculates and returns the EMA of entropy.

        Returns:
            float: EMA of entropy.
        """
        entropy = super().value
        self._update_ema(ticker='entropy', entropy=entropy)
        return self.entropy_ema.get('entropy', np.nan)
