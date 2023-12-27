"""
This module defines two classes, TradeFlowMonitor and TradeFlowEMAMonitor, for monitoring the net trade volume flow of a given underlying.

Classes:
- TradeFlowMonitor: Monitors net trade volume flow for each underlying and provides the values.
- TradeFlowEMAMonitor: Extends TradeFlowMonitor and adds Exponential Moving Averages (EMA) to trade flow and trade volume.

Usage:
1. Instantiate the desired monitor class with optional parameters.
2. Call the instance with market data to update the monitor.
3. Retrieve the calculated values using the 'value' property.

Note: This module assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-27
"""

from AlgoEngine.Engine import MarketDataMonitor
from PyQuantKit import MarketData, TradeData

from .. import EMA, Synthetic, MDS


class TradeFlowMonitor(MarketDataMonitor):
    """
    Monitors net trade volume flow for each underlying and provides the values, with unit of share.

    Args:
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.TradeFlow'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, name: str = 'Monitor.TradeFlow', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, mds=MDS)

        self._trade_flow = dict()
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the monitor based on market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        if isinstance(market_data, TradeData):
            return self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData):
        """
        Handle trade data to update net trade volume flow.

        Args:
        - trade_data (TradeData): Trade data to handle.
        """
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign

        self._trade_flow[ticker] = self._trade_flow.get(ticker, 0.) + volume * side

    def clear(self):
        """Clear the monitor data."""
        self._trade_flow.clear()

    @property
    def value(self) -> dict[str, float]:
        """
        Get the net trade volume flow values.

        Returns:
        dict[str, float]: Dictionary of net trade volume flow values.
        """
        return self._trade_flow

    @property
    def is_ready(self) -> bool:
        """
        Check if the monitor is ready.

        Returns:
        bool: True if the monitor is ready, False otherwise.
        """
        return self._is_ready


class TradeFlowEMAMonitor(TradeFlowMonitor, EMA, Synthetic):
    """
    Exponential Moving Average (EMA) of net trade volume flow.

    Args:
    - discount_interval (float): Interval for EMA discounting.
    - alpha (float): EMA smoothing factor.
    - weights (dict[str, float]): Dictionary of ticker weights.
    - normalized (bool, optional): Whether to use normalized EMA. Defaults to True.
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.TradeFlow.EMA'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, discount_interval: float, alpha: float, weights: dict[str, float], normalized: bool = True, name: str = 'Monitor.TradeFlow.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)
        Synthetic.__init__(self=self, weights=weights)

        self.normalized = normalized

        self._trade_flow = self._register_ema(name='trade_flow')
        self._trade_volume = self._register_ema(name='trade_volume')

    def _on_trade(self, trade_data: TradeData):
        """
        Handle trade data to update net trade volume flow and accumulate EMAs.

        Args:
        - trade_data (TradeData): Trade data to handle.
        """
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        self._accumulate_ema(ticker=ticker, timestamp=timestamp, trade_flow=volume * side, trade_volume=volume)

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the monitor and trigger EMA discounting based on market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self._discount_ema(ticker=ticker, timestamp=timestamp)
        self._discount_all(timestamp=timestamp)

        super().__call__(market_data=market_data, **kwargs)

    def clear(self):
        """Clear the monitor data."""
        super().clear()
        EMA.clear(self)
        Synthetic.clear(self)

        self._trade_flow = self._register_ema(name='trade_flow')
        self._trade_volume = self._register_ema(name='trade_volume')

    def trade_flow_adjusted(self):
        """
        Get adjusted net trade volume flow values.

        Returns:
        dict[str, float]: Dictionary of adjusted net trade volume flow values.
        """
        if self.normalized:
            normalized_trade_flow = {}

            for ticker in self._trade_flow:
                net_flow = self._trade_flow[ticker]
                volume = self._trade_volume[ticker]
                adjusted_flow = net_flow / volume if volume else 0.

                normalized_trade_flow[ticker] = adjusted_flow

                if abs(adjusted_flow) > 1:
                    raise ValueError('adjusted_flow should not be larger than 1 or smaller than -1, check the code!')

            return normalized_trade_flow
        else:
            return super().value

    @property
    def value(self) -> dict[str, float]:
        """
        Get the adjusted values of net trade volume flow and the composite index value.

        Returns:
        dict[str, float]: Dictionary of adjusted values.
        """
        result = {}
        result.update(self.trade_flow_adjusted())
        result['Index'] = self.index_value
        return result

    @property
    def index_value(self) -> float:
        """
        Get the composite index value.

        Returns:
        float: Composite index value.
        """
        return self.composite(self.value)
