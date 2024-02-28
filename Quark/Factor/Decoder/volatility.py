"""
This module defines a VolatilityMonitor class for monitoring market data and calculating a weighted index based on daily volatility.

Classes:
- VolatilityMonitor: Monitors market data for daily volatility and calculates a weighted index.

Usage:
1. Instantiate the VolatilityMonitor with weights and optional name and monitor_id.
2. Call the instance with market data to update the monitor.
3. Retrieve the calculated values using the 'value' and 'weighted_index' properties.
4. Clear the monitor data using the 'clear' method when needed.

Note: This module assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-27
"""
from __future__ import annotations

import json

import numpy as np
from PyQuantKit import MarketData

from .. import Synthetic, FactorMonitor


class VolatilityMonitor(FactorMonitor, Synthetic):
    """
    Class for monitoring market data and calculating a weighted index based on daily volatility.

    Args:
    - weights (dict[str, float]): Dictionary of ticker weights.
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.Volatility.Daily'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, weights: dict[str, float], name: str = 'Monitor.Volatility.Daily', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

        # External data that must be assigned
        self.daily_volatility: dict[str, float] = {}  # must be assigned from outside
        self.index_volatility: float = np.nan  # must be assigned from outside

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the synthetic index based on the received market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        self.update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            daily_volatility=self.daily_volatility,
            index_volatility=self.index_volatility
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> VolatilityMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            weights=json_dict['weights'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        """
        Clear the monitor data, including daily volatility and synthetic data.
        """
        self.daily_volatility.clear()
        self.index_volatility = np.nan
        Synthetic.clear(self)

    def volatility_adjusted(self) -> dict[str, float]:
        """
        Calculate and return volatility-adjusted values for each ticker.

        Returns:
        dict[str, float]: Dictionary of volatility-adjusted values for each ticker.
        """
        volatility_adjusted = {}

        for ticker in self.weights:
            if ticker not in self.daily_volatility:
                continue

            if ticker not in self.last_price:
                continue

            volatility_adjusted[ticker] = (self.last_price[ticker] / self.base_price[ticker] - 1) / self.daily_volatility[ticker]

        return volatility_adjusted

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index'
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    @property
    def value(self) -> dict[str, float]:
        """
        Calculate and return volatility-adjusted values for each ticker and the weighted index.

        Returns:
        dict[str, float]: Dictionary of volatility-adjusted values for each ticker and the weighted index.
        """
        result = self.volatility_adjusted()
        result['Index'] = self.weighted_index
        return result

    @property
    def weighted_index(self) -> float:
        """
        Calculate and return the weighted index based on volatility-adjusted values and weights.

        Returns:
        float: Weighted index value.
        """
        volatility_adjusted = self.volatility_adjusted()
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
