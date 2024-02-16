"""
This module defines two classes, AggressivenessMonitor and AggressivenessEMAMonitor, for monitoring the aggressiveness of buy/sell trades.

Classes:
- AggressivenessMonitor: Monitors the aggressiveness of buy/sell trades and provides aggregated values.
- AggressivenessEMAMonitor: Extends AggressivenessMonitor and adds Exponential Moving Averages (EMA) to aggressive buying, selling, and total trade volumes.

Usage:
1. Instantiate the desired monitor class with optional parameters.
2. Call the instance with market data to update the monitor.
3. Retrieve the calculated values using the 'value' property.

Note: This module assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2023-12-27
"""
from __future__ import annotations

import json

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData

from .. import EMA, Synthetic, FactorMonitor, DEBUG_MODE


class AggressivenessMonitor(FactorMonitor):
    """
    Monitors the aggressiveness of buy/sell trades and provides aggregated values.

    Args:
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.Aggressiveness'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, name: str = 'Monitor.Aggressiveness', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)

        self._last_update: dict[str, float] = dict()
        self._trade_price: dict[str, dict[int, float]] = dict()
        self._aggressive_buy: dict[str, float] = {}
        self._aggressive_sell: dict[str, float] = {}
        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the monitor based on market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        if isinstance(market_data, (TradeData, TransactionData)):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData | TransactionData):
        """
        Handle trade data to update aggressive buying/selling volumes.

        Args:
        - trade_data : Trade data to handle.
        """
        ticker = trade_data.ticker
        price = trade_data.market_price
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        if ticker in self._trade_price:
            trade_price_log = self._trade_price[ticker]
        else:
            trade_price_log = self._trade_price[ticker] = {}

        if ticker not in self._last_update or self._last_update[ticker] < timestamp:
            trade_price_log.clear()
            self._last_update[ticker] = timestamp
            return

        if side > 0:  # a buy init trade
            order_id: int = trade_data.buy_id
            if order_id is None:
                self._is_ready = False
        elif side < 0:
            order_id: int = trade_data.sell_id
            if order_id is None:
                self._is_ready = False
        else:
            return

        if order_id not in trade_price_log:
            trade_price_log[order_id] = price
        else:
            last_price = trade_price_log[order_id]

            if last_price == price:
                pass
            else:
                self._update_aggressiveness(
                    ticker=ticker,
                    side=side,
                    volume=volume,
                    timestamp=timestamp
                )

        self._last_update[ticker] = timestamp

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        """
        Update aggressive buying/selling volumes.

        Args:
        - ticker (str): Ticker symbol.
        - volume (float): Trade volume.
        - side (int): Trade side (-1 for sell, 1 for buy).
        - timestamp (float): Trade timestamp.
        """
        if side > 0:
            self._aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) + volume
        else:
            self._aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) + volume

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            last_update=self._last_update,
            trade_price=self._trade_price,
            aggressive_buy=self._aggressive_buy,
            aggressive_sell=self._aggressive_sell,
            is_ready=self._is_ready
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> AggressivenessMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._last_update = json_dict['last_update']
        self._trade_price = json_dict['trade_price']
        self._aggressive_buy = json_dict['aggressive_buy']
        self._aggressive_sell = json_dict['aggressive_sell']
        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        """Clear the monitor data."""
        self._last_update.clear()
        self._trade_price.clear()
        self._aggressive_buy.clear()
        self._aggressive_sell.clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Net'
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.Buy' for ticker in subscription
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.Sell' for ticker in subscription
        ]

    @property
    def value(self) -> dict[str, float]:
        """
        Get the aggregated values of aggressive buying/selling volumes.

        Returns:
        dict[str, float]: Dictionary of aggregated values.
        """
        result = {}

        for ticker in set(self._aggressive_buy) | set(self._aggressive_sell):
            result[f'{ticker}.Buy'] = self._aggressive_buy.get(ticker, 0.)
            result[f'{ticker}.Sell'] = self._aggressive_sell.get(ticker, 0.)

        result['Net'] = np.nansum(list(self._aggressive_buy.values())) - np.nansum(list(self._aggressive_sell.values()))
        return result

    @property
    def is_ready(self) -> bool:
        """
        Check if the monitor is ready.

        Returns:
        bool: True if the monitor is ready, False otherwise.
        """
        return self._is_ready


class AggressivenessEMAMonitor(AggressivenessMonitor, EMA, Synthetic):
    """
    Exponential Moving Average (EMA) of aggressive buying/selling volume.

    Args:
    - discount_interval (float): Interval for EMA discounting.
    - alpha (float): EMA smoothing factor.
    - weights (dict[str, float]): Dictionary of ticker weights.
    - normalized (bool, optional): Whether to use normalized EMA. Defaults to True.
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.Aggressiveness.EMA'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, discount_interval: float, alpha: float, weights: dict[str, float], normalized: bool = True, name: str = 'Monitor.Aggressiveness.EMA', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, discount_interval=discount_interval, alpha=alpha)
        Synthetic.__init__(self=self, weights=weights)

        self.normalized = normalized

        self._aggressive_buy: dict[str, float] = self.register_ema(name='aggressive_buy')  # EMA of aggressive buying volume
        self._aggressive_sell: dict[str, float] = self.register_ema(name='aggressive_sell')  # EMA of aggressive selling volume
        self._trade_volume: dict[str, float] = self.register_ema(name='trade_volume')  # EMA of total trade volume

    def _update_aggressiveness(self, ticker: str, volume: float, side: int, timestamp: float):
        """
        Update aggressive buying/selling volumes and accumulate EMAs.

        Args:
        - ticker (str): Ticker symbol.
        - volume (float): Trade volume.
        - side (int): Trade side (-1 for sell, 1 for buy).
        - timestamp (float): Trade timestamp.
        """
        if side > 0:
            self.accumulate_ema(ticker=ticker, timestamp=timestamp, replace_na=0., aggressive_buy=volume, aggressive_sell=0.)
        else:
            self.accumulate_ema(ticker=ticker, timestamp=timestamp, replace_na=0., aggressive_buy=0., aggressive_sell=volume)

        if DEBUG_MODE:
            total_volume = self._trade_volume[ticker]
            aggressiveness_volume = self._aggressive_buy[ticker] if side > 0 else self._aggressive_sell[ticker]
            adjusted_volume_ratio = aggressiveness_volume / total_volume if total_volume else 0.

            if not (0 <= adjusted_volume_ratio <= 1):
                raise ValueError(f'{ticker} {self.__class__.__name__} encounter invalid value, total_volume={total_volume}, aggressiveness={aggressiveness_volume}, side={side}')

    def _on_trade(self, trade_data: TradeData | TransactionData):
        """
        Handle trade data to update aggressive buying/selling volumes and accumulate EMAs.

        Args:
        - trade_data: Trade data to handle.
        """
        ticker = trade_data.ticker
        self.accumulate_ema(ticker=ticker, replace_na=0., trade_volume=trade_data.volume)  # to avoid redundant calculation, timestamp is not passed-in, so that the discount function will not be triggered
        super()._on_trade(trade_data=trade_data)

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the monitor and trigger EMA discounting based on market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp

        self.discount_ema(ticker=ticker, timestamp=timestamp)
        self.discount_all(timestamp=timestamp)

        return super().__call__(market_data=market_data, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            normalized=self.normalized
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> AggressivenessEMAMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            discount_interval=json_dict['discount_interval'],
            alpha=json_dict['alpha'],
            weights=json_dict['weights'],
            normalized=json_dict['normalized'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._last_update = json_dict['last_update']
        self._trade_price = json_dict['trade_price']
        self._aggressive_buy = json_dict['aggressive_buy']
        self._aggressive_sell = json_dict['aggressive_sell']

        self._aggressive_buy = self.ema['aggressive_buy']
        self._aggressive_sell = self.ema['aggressive_sell']
        self._trade_volume = self.ema['trade_volume']

        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        """Clear the monitor data."""
        super().clear()
        EMA.clear(self)
        Synthetic.clear(self)

        self._aggressive_buy: dict[str, float] = self.register_ema(name='aggressive_buy')  # EMA of aggressive buying volume
        self._aggressive_sell: dict[str, float] = self.register_ema(name='aggressive_sell')  # EMA of aggressive selling volume
        self._trade_volume: dict[str, float] = self.register_ema(name='trade_volume')  # EMA of total trade volume

    def aggressiveness_adjusted(self):
        """
        Get adjusted aggressive buying/selling volumes.

        Returns:
        tuple[dict[str, float], dict[str, float]]: Adjusted aggressive buying and selling volumes.
        """
        aggressive_buy = {}
        aggressive_sell = {}

        for ticker, volume in self._trade_volume.items():
            if volume:
                aggressive_buy[ticker] = self._aggressive_buy.get(ticker, 0.) / volume
                aggressive_sell[ticker] = self._aggressive_sell.get(ticker, 0.) / volume

        return aggressive_buy, aggressive_sell

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index'
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.Buy' for ticker in subscription
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}.Sell' for ticker in subscription
        ]

    @property
    def value(self) -> dict[str, float]:
        """
        Get the adjusted values of aggressive buying/selling volumes and the composite index value.

        Returns:
        dict[str, float]: Dictionary of adjusted values.
        """
        result = {}

        for ticker, volume in self._trade_volume.items():
            if volume:
                result[f'{ticker}.Buy'] = self._aggressive_buy.get(ticker, 0.) / volume
                result[f'{ticker}.Sell'] = self._aggressive_sell.get(ticker, 0.) / volume

        result['Index'] = self.index_value

        return result

    @property
    def index_value(self) -> float:
        """
        Get the composite index value.

        Returns:
        float: Composite index value.
        """
        aggressive_buy, aggressive_sell = self.aggressiveness_adjusted()
        values = {}

        for ticker in set(aggressive_buy) | set(aggressive_sell):
            values[ticker] = aggressive_buy.get(ticker, 0) - aggressive_sell.get(ticker, 0.)

        return self.composite(values)
