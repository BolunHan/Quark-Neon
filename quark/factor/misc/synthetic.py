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

import json
from typing import Self

from algo_engine.base import MarketData, TradeData, TransactionData, BarData

from .. import Synthetic, FactorMonitor, FixedIntervalSampler, SamplerMode


class SyntheticIndexMonitor(FactorMonitor, FixedIntervalSampler, Synthetic):
    """
    Monitors market data and generates synthetic bar data for index price and volume movement.

    Note: the name of this class is used in collect_factor function, do not amend the name!

    Args:
    - index_name (str): Name of the synthetic index.
    - weights (dict[str, float]): Dictionary of ticker weights.
    - interval (float, optional): Interval for synthetic bar data. Defaults to 60.
    - name (str, optional): Name of the monitor. Defaults to 'Monitor.SyntheticIndex'.
    - monitor_id (str, optional): Identifier for the monitor. Defaults to None.
    """

    def __init__(self, index_name: str, weights: dict[str, float], sampling_interval: float = 60., name='Monitor.SyntheticIndex', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=1)
        Synthetic.__init__(self=self, weights=weights)

        self.register_sampler(topic='open_price', mode=SamplerMode.first)
        self.register_sampler(topic='close_price', mode=SamplerMode.update)
        self.register_sampler(topic='high_price', mode=SamplerMode.max)
        self.register_sampler(topic='low_price', mode=SamplerMode.min)
        self.register_sampler(topic='volume', mode=SamplerMode.accumulate)
        self.register_sampler(topic='notional', mode=SamplerMode.accumulate)
        self.register_sampler(topic='flow', mode=SamplerMode.accumulate)
        self.register_sampler(topic='trade_count', mode=SamplerMode.accumulate)

        self.index_name = index_name
        self.timestamp = 0
        self._serializable = True

    def __call__(self, market_data: MarketData, allow_out_session: bool = True, **kwargs):
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        super().__call__(market_data, allow_out_session=allow_out_session, **kwargs)

    def on_market_data(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        price = market_data.market_price

        if isinstance(market_data, (TradeData, TransactionData)):
            volume = market_data.volume
            notional = market_data.notional
            flow = market_data.flow
        else:
            volume = notional = flow = 0

        self.log_obs(
            ticker=ticker,
            timestamp=timestamp,
            open_price=price,
            close_price=price,
            high_price=price,
            low_price=price,
            volume=volume,
            notional=notional,
            flow=flow,
            trade_count=1
        )

        self.timestamp = timestamp

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            index_name=self.index_name
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            index_name=json_dict['index_name'],
            weights=json_dict['weights'],
            sampling_interval=json_dict['sampling_interval'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)
        return self

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
    def is_ready(self) -> bool:
        return bool(self.base_price)

    @property
    def value(self) -> dict[str, float]:
        """
        Retrieve the last generated bar data.

        Returns:
        dict[str, float]: Dictionary of last generated bar data.
        """

        base_price = self.composite(self.base_price) if self.base_price else 1.
        open_price = self.composite(self.get_active(topic='open_price')) / base_price
        close_price = self.composite(self.get_latest(topic='close_price')) / base_price
        high_price = self.composite(self.get_latest(topic='high_price')) / base_price
        low_price = self.composite(self.get_latest(topic='low_price')) / base_price
        volume = sum(self.get_latest(topic='volume'))
        notional = sum(self.get_latest(topic='notional'))
        # flow = sum(self.get_latest(topic='flow'))
        trade_count = sum(self.get_latest(topic='trade_count'))

        result = dict(
            market_price=self.index_price,
            high_price=high_price,
            low_price=low_price,
            open_price=open_price,
            close_price=close_price,
            volume=volume,
            notional=notional,
            trade_count=trade_count
        )

        return result

    @property
    def index_price(self) -> float:
        """
        return cached index price, if not hit, generate new one.
        Returns: (float) the synthetic index price

        """
        return self.synthetic_index

    @property
    def active_bar(self) -> BarData | None:
        """
        Retrieve the currently active bar data.

        Returns:
        BarData | None: Currently active bar data.
        """

        if not self.is_ready:
            return None

        candlestick = BarData(
            ticker=self.index_name,
            timestamp=self.timestamp,
            **self.value
        )

        return candlestick

    @property
    def last_bar(self) -> BarData | None:
        """
        Retrieve the last bar data.

        Returns:
        BarData | None: last bar data.
        """
        base_price = self.composite(self.base_price) if self.base_price else 1.
        open_price = self.composite({ticker: storage[-1] for ticker, storage in self.get_history(topic='open_price') if storage}) / base_price
        close_price = self.composite({ticker: storage[-1] for ticker, storage in self.get_history(topic='close_price') if storage}) / base_price
        high_price = self.composite({ticker: storage[-1] for ticker, storage in self.get_history(topic='high_price') if storage}) / base_price
        low_price = self.composite({ticker: storage[-1] for ticker, storage in self.get_history(topic='low_price') if storage}) / base_price
        volume = sum([storage[-1] for storage in self.get_history(topic='volume').values() if storage])
        notional = sum([storage[-1] for storage in self.get_history(topic='notional').values() if storage])
        flow = sum([storage[-1] for storage in self.get_history(topic='flow').values() if storage])
        trade_count = sum([storage[-1] for storage in self.get_history(topic='trade_count').values() if storage])

        candlestick = BarData(
            ticker=self.index_name,
            timestamp=self.timestamp,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
            notional=notional,
            flow=flow,
            trade_count=trade_count
        )

        return candlestick

    @property
    def serializable(self) -> bool:
        return self._serializable

    @serializable.setter
    def serializable(self, serializable: bool) -> None:
        self._serializable = serializable
