"""
This module defines a SyntheticIndexMonitor class for synthesizing index price/volume movement based on market data.

Classes:
- SyntheticIndexMonitor: Monitors market data and generates synthetic bar data for index price/volume movement.

Usage:
1. Instantiate the SyntheticIndexMonitor with an index name, weights, and optional interval, name, and monitor_id.
2. Call the instance with market data to update the monitor and generate synthetic bar data.
3. Retrieve the last generated bar data using the 'value' property.
4. Retrieve the synthetic index price using the 'index_price' property.
5. Retrieve the currently active bar data using the 'active_bar' property.
6. Clear the monitor data using the 'clear' method when needed.

Note: This module assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-27
"""

import datetime

from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData, BarData

from .. import Synthetic, MDS, TIME_ZONE


class SyntheticIndexMonitor(MarketDataMonitor, Synthetic):
    """
    Monitors market data and generates synthetic bar data for index price and volume movement.

    Args:
    - index_name (str): Name of the synthetic index.
    - weights (dict[str, float]): Dictionary of ticker weights.
    - interval (float, optional): Interval for synthetic bar data. Defaults to 60.
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.SyntheticIndex'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, index_name: str, weights: dict[str, float], interval: float = 60., name='Monitor.SyntheticIndex', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)
        Synthetic.__init__(self=self, weights=weights)

        self.index_name = index_name
        self.interval = interval

        self._active_bar_data: BarData | None = None
        self._last_bar_data: BarData | None = None

        self._is_ready = True
        self._value = {}

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the synthetic index and generate synthetic bar data based on received market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        self._update_synthetic(ticker=ticker, market_price=market_price)

        if ticker not in self.weights:
            return

        index_price = self.synthetic_index

        if self._active_bar_data is None or timestamp >= self._active_bar_data.timestamp:
            self._last_bar_data = self._active_bar_data
            bar_data = self._active_bar_data = BarData(
                ticker=self.index_name,
                bar_start_time=datetime.datetime.fromtimestamp(timestamp // self.interval * self.interval, tz=TIME_ZONE),
                timestamp=(timestamp // self.interval + 1) * self.interval,  # by definition, timestamp when the bar ends
                bar_span=datetime.timedelta(seconds=self.interval),
                high_price=index_price,
                low_price=index_price,
                open_price=index_price,
                close_price=index_price,
                volume=0.,
                notional=0.,
                trade_count=0
            )
        else:
            bar_data = self._active_bar_data

        if isinstance(market_data, TradeData):
            bar_data.volume += market_data.volume
            bar_data.notional += market_data.notional
            bar_data.trade_count += 1

        bar_data.close_price = index_price
        bar_data.high_price = max(bar_data.high_price, index_price)
        bar_data.low_price = min(bar_data.low_price, index_price)

    def clear(self):
        """
        Clear the monitor data, including bar data and values.
        """
        self._active_bar_data = None
        self._last_bar_data = None
        self._value.clear()

    @property
    def is_ready(self) -> bool:
        """
        Check if the monitor is ready based on the availability of the last bar data.

        Returns:
        bool: True if the monitor is ready, False otherwise.
        """
        if self._last_bar_data is None:
            return False
        else:
            return self._is_ready

    @property
    def value(self) -> BarData | None:
        """
        Retrieve the last generated bar data.

        Returns:
        BarData | None: Last generated bar data.
        """
        return self._last_bar_data

    @property
    def index_price(self) -> float:
        """
        Retrieve the synthetic index price.

        Returns:
        float: Synthetic index price.
        """
        return self.synthetic_index

    @property
    def active_bar(self) -> BarData | None:
        """
        Retrieve the currently active bar data.

        Returns:
        BarData | None: Currently active bar data.
        """
        return self._active_bar_data
