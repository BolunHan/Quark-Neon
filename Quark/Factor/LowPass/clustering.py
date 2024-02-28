from __future__ import annotations

import json
import warnings
from typing import Self

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller

from .. import FactorMonitor, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, Synthetic, EMA


class TradeClusteringMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, name: str = 'Monitor.Trade.Clustering', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price')
        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='volume_net', mode='accumulate')
        self.register_sampler(name='trade_imbalance')

        self._unit_root: dict[str, float] = {}

        warnings.simplefilter('ignore', InterpolationWarning)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        timestamp = market_data.timestamp
        market_price = market_data.market_price

        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

        if isinstance(market_data, (TradeData, TransactionData)):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData | TransactionData):
        """
        Updates volume and net volume based on trade data.

        Args:
            trade_data: Trade data object containing volume and side information.
        """
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        self.log_obs(ticker=ticker, timestamp=timestamp, volume=volume, volume_net=volume * side)

        if total_volume := self.active_obs(name='volume')[ticker]:
            trade_imbalance = self.active_obs(name='volume_net').get(ticker, 0.) / total_volume
            self.log_obs(ticker=ticker, timestamp=timestamp, trade_imbalance=trade_imbalance)

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'trade_imbalance':
            return

        if not self.is_ready:
            return

        volume_flow = np.array(list(self.get_sampler(name='trade_imbalance')[ticker]))
        volume_flow = volume_flow[:-1]  # remove the active entry

        adf_value, p_value, lag, *_ = adfuller(
            x=volume_flow,
            maxlag=int(len(volume_flow) / 2 - 5),
            regression='ct'
        )

        # kpss_value, p_value, lag, *_ = kpss(
        #     x=volume_flow,
        #     nlags='auto',
        #     regression='ct'
        # )

        self._unit_root[ticker] = EMA.calculate_ema(value=p_value, memory=self._unit_root.get(ticker), alpha=0.5)
        self._unit_root[ticker] = p_value

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            unit_root=self._unit_root
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def update_from_json(self, json_dict: dict) -> Self:
        super().update_from_json(json_dict=json_dict)
        self._unit_root.update(json_dict['unit_root'])
        return self

    def clear(self):
        FixedIntervalSampler.clear(self)

        self._unit_root.clear()

        self.register_sampler(name='price')
        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='volume_net', mode='accumulate')
        self.register_sampler(name='trade_imbalance')

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    @property
    def value(self) -> dict[str, float]:
        return self._unit_root


class TradeClusteringAdaptiveMonitor(TradeClusteringMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 15, aligned_interval: bool = False, name: str = 'Monitor.Trade.Clustering.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval
        )

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        AdaptiveVolumeIntervalSampler.clear(self)

        super().clear()

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class TradeClusteringIndexAdaptiveMonitor(TradeClusteringAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 15, weights: dict[str, float] = None, aligned_interval: bool = True, name: str = 'Monitor.Trade.Clustering.Index.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            baseline_window=baseline_window,
            sample_size=sample_size,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        super().__call__(market_data=market_data, **kwargs)

    def _on_trade(self, trade_data: TradeData | TransactionData):
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        self.log_obs(ticker=ticker, timestamp=timestamp, volume=volume, volume_net=volume * side)

        if total_volume := self.active_obs(name='volume')[ticker]:
            trade_imbalance = self.active_obs(name='volume_net').get(ticker, 0.) / total_volume
            self.log_obs(ticker=ticker, timestamp=timestamp, trade_imbalance=trade_imbalance)

            self.accumulate_volume(ticker='Synthetic', volume=trade_data.notional)
            self.update_synthetic(ticker=ticker, market_price=trade_imbalance)
            self.log_obs(ticker='Synthetic', timestamp=trade_data.timestamp, trade_imbalance=self.synthetic_index)

    def clear(self):
        Synthetic.clear(self)

        super().clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index',
            f'{self.name.removeprefix("Monitor.")}.Weighted',
        ]

    @property
    def value(self) -> dict[str, float]:
        return {'Index': self._unit_root.get('Synthetic', np.nan), 'Weighted': self.composite(self._unit_root)}
