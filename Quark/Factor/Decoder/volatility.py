import numpy as np
from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData

from .. import MDS, Synthetic


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

        index_volatility_range = (self.synthetic_index / self.synthetic_base_price - 1) / weighted_volatility

        if not index_volatility_range:
            return 0.

        weighted_index -= index_volatility_range
        weighted_index -= diff_base

        return weighted_index

    @property
    def is_ready(self) -> bool:
        return self._is_ready
