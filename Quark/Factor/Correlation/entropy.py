import numpy as np
from PyQuantKit import MarketData

from .coherence import CoherenceMonitor
from .. import EMA


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
