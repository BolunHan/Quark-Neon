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
from __future__ import annotations

import datetime
import json

from PyQuantKit import MarketData, TradeData, TransactionData, BarData

from .. import Synthetic, FactorMonitor, TIME_ZONE


class SyntheticIndexMonitor(FactorMonitor, Synthetic):
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
        super().__init__(name=name, monitor_id=monitor_id)
        Synthetic.__init__(self=self, weights=weights)

        self.index_name = index_name
        self.interval = interval

        self._active_bar_data: BarData | None = None
        self._last_bar_data: BarData | None = None
        self._index_price = None

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        """
        Update the synthetic index and generate synthetic bar data based on received market data.

        Args:
        - market_data (MarketData): Market data to update the monitor.
        """
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        if ticker in self.weights:
            self.update_synthetic(ticker=ticker, market_price=market_price)
            self._index_price = index_price = self.synthetic_index
        elif ticker == self.index_name:
            self._index_price = index_price = market_price
        else:
            return

        if self._active_bar_data is None or timestamp >= self._active_bar_data.timestamp:
            self._last_bar_data = self._active_bar_data
            bar_data = self._active_bar_data = BarData(
                ticker=self.index_name,
                bar_start_time=datetime.datetime.fromtimestamp((timestamp // self.interval) * self.interval, tz=TIME_ZONE),
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

        if isinstance(market_data, (TradeData, TransactionData)):
            bar_data['volume'] += market_data.volume
            bar_data['notional'] += market_data.notional
            bar_data['trade_count'] += 1

        bar_data['close_price'] = index_price
        bar_data['high_price'] = max(bar_data.high_price, index_price)
        bar_data['low_price'] = min(bar_data.low_price, index_price)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            index_name=self.index_name,
            interval=self.interval,
            active_bar_data=self._active_bar_data.to_json(fmt='dict') if self._active_bar_data is not None else None,
            last_bar_data=self._last_bar_data.to_json(fmt='dict') if self._last_bar_data is not None else None,
            index_price=self._index_price,
            is_ready=self._is_ready
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> SyntheticIndexMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            index_name=json_dict['index_name'],
            weights=json_dict['weights'],
            interval=json_dict['interval'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._active_bar_data = BarData.from_json(json_dict['active_bar_data']) if json_dict['active_bar_data'] is not None else None
        self._last_bar_data = BarData.from_json(json_dict['last_bar_data']) if json_dict['last_bar_data'] is not None else None
        self._index_price = json_dict['index_price']
        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        """
        Clear the monitor data, including bar data and values.
        """
        self._active_bar_data = None
        self._last_bar_data = None

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

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.market_price',
            f'{self.name.removeprefix("Monitor.")}.high_price',
            f'{self.name.removeprefix("Monitor.")}.low_price',
            f'{self.name.removeprefix("Monitor.")}.open_price',
            f'{self.name.removeprefix("Monitor.")}.close_price',
            f'{self.name.removeprefix("Monitor.")}.volume',
            f'{self.name.removeprefix("Monitor.")}.notional',
            f'{self.name.removeprefix("Monitor.")}.trade_count',
        ]

    @property
    def value(self) -> dict[str, float]:
        """
        Retrieve the last generated bar data.

        Returns:
        dict[str, float]: Dictionary of last generated bar data.
        """
        bar_data = self._last_bar_data

        if bar_data is not None:
            result = dict(
                market_price=self.index_price,
                high_price=bar_data.high_price,
                low_price=bar_data.low_price,
                open_price=bar_data.open_price,
                close_price=bar_data.close_price,
                volume=bar_data.volume,
                notional=bar_data.notional,
                trade_count=bar_data.trade_count
            )
        else:
            result = dict(
                market_price=self.index_price
            )

        return result

    @property
    def index_price(self) -> float:
        """
        return cached index price, if not hit, generate new one.
        Returns: (float) the synthetic index price

        """
        if self._index_price is None:
            return self.synthetic_index
        else:
            return self._index_price

    @property
    def active_bar(self) -> BarData | None:
        """
        Retrieve the currently active bar data.

        Returns:
        BarData | None: Currently active bar data.
        """
        return self._active_bar_data

    @property
    def last_bar(self) -> BarData | None:
        """
        Retrieve the last bar data.

        Returns:
        BarData | None: last bar data.
        """
        return self._last_bar_data
