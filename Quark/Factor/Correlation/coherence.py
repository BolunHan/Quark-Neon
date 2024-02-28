"""
This script defines classes for monitoring and measuring the coherence of price and volume percentage changes
in a stock pool. It utilizes market and trade data from PyQuantKit and includes functionalities for calculating
dispersion coefficients, exponential moving averages (EMA), and regression slopes.

Classes:
- CoherenceMonitor: Monitors and measures the coherence of price percentage change.
- CoherenceAdaptiveMonitor: Extends CoherenceMonitor and includes adaptive volume sampling.
- CoherenceEMAMonitor: Extends CoherenceMonitor and includes an EMA for dispersion ratio.
- TradeCoherenceMonitor: Monitors and measures the coherence of volume percentage change based on trade data.

Usage:
1. Instantiate the desired monitor class with appropriate parameters.
2. Call the instance with market or trade data to update the monitor.
3. Retrieve the coherence values using the 'value' property of the monitor instance.

Note: This script assumes the availability of AlgoEngine, PyQuantKit, and other required modules.

Author: Bolun
Date: 2024-01-25
"""
from __future__ import annotations

import json
import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression

from .. import FactorMonitor, EMA, FixedIntervalSampler, AdaptiveVolumeIntervalSampler


class CoherenceMonitor(FactorMonitor, FixedIntervalSampler):
    """
    Monitors and measures the coherence of price percentage change.

    The correlation coefficient of the regression log(y)~rank(x) is usually above 90%
    The slope of the upper part of the price pct change distribution is called dispersion
    A low up dispersion (< 0.15) or a high dispersion ratio indicate a turning point of upward trend
    A low down dispersion (< 0.15) or a low dispersion ratio indicate a turning point of downward trend

    Attributes:
        sampling_interval (float): Time interval for sampling market data.
        sample_size (float): max sample size
        weights (dict): Weights for individual stocks in the pool.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
    """

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float] = None, center_mode='median', name: str = 'Monitor.Coherence.PricePct', monitor_id: str = None):
        """
        Initializes the CoherenceMonitor.

        Args:
            sampling_interval (float): Time interval for sampling market data.
            sample_size (float): Max sample size for data collection.
            weights (dict): Weights for individual stocks in the pool.
            center_mode (str): Mode for calculating center in dispersion.
            name (str): Name of the monitor.
            monitor_id (str): Identifier for the monitor.
        """
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.weights = weights
        self.center_mode = center_mode

        self.register_sampler(name='price', mode='update')

    def __call__(self, market_data: MarketData, **kwargs):

        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

    @classmethod
    def slope(cls, y: list[float] | np.ndarray, x: list[float] | np.ndarray = None):
        x = np.array(x)
        x = np.vstack([x, np.ones(len(x))]).T
        y = np.array(y)

        slope, c = np.linalg.lstsq(x, y, rcond=None)[0]

        return slope

    def dispersion(self, side: int, center_mode: str = 'absolute'):
        """
        Collects dispersion data based on price change.

        Args:
            side (int): Sign indicating upward (1) or downward (-1) trend.
            center_mode (str): Mode for calculating center in dispersion.

        Returns:
            float: Dispersion coefficient.
        """
        historical_price = self.get_sampler(name='price')
        price_pct_matrix = []
        weights = []

        if _ := [len(historical_price[ticker]) for ticker in historical_price if not (self.weights and ticker not in self.weights)]:
            vector_length = min(_)
        else:
            return np.nan

        if vector_length < 3:
            return np.nan

        for ticker in historical_price:
            if self.weights and ticker not in self.weights:
                continue

            price_vector = np.array(list(historical_price[ticker])[-vector_length:])
            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            price_pct_matrix.append(price_pct_vector)
            weights.append(self.weights[ticker] if self.weights else 1)

        # to data snapshot
        values = []
        for price_pct_distribution in np.array(price_pct_matrix).T:
            if center_mode == 'absolute':
                center = 0.
            elif center_mode == 'median':
                center = np.median(price_pct_distribution)
            elif center_mode == 'mean':
                center = np.mean(price_pct_distribution)
            elif center_mode == 'weighted':
                center = np.average(price_pct_distribution, weights=weights)
            else:
                raise NotImplementedError(f'Invalid center mode {center_mode}. Expect absolute, median, mean or weighted.')

            y_selected = []
            weights_selected = []

            for weight, price_change in zip(weights, price_pct_distribution):
                if (price_change - center) * side <= 0:  # for entries with value == center, they will be excluded from both side
                    continue

                y_selected.append(abs(price_change - center))
                weights_selected.append(weight)

            x = rankdata(y_selected)
            y = np.log(y_selected)

            if len(y) < 3:
                continue

            x = np.vstack([x, np.ones(len(x))]).T

            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X=x, y=y, sample_weight=weights_selected)
            # regressor.fit(X=x, y=y)
            slope = regressor.coef_[0]
            # y_pred = regressor.predict(x)
            # r2 = r2_score(y_true=y, y_pred=y_pred, sample_weight=weights_selected)
            values.append(slope)

        return np.nanmean(values)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            weights=dict(self.weights),
            center_mode=self.center_mode
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> CoherenceMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            weights=json_dict['weights'],
            center_mode=json_dict['center_mode'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        return self

    def clear(self):
        FixedIntervalSampler.clear(self)

        self.register_sampler(name='price', mode='update')

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.up',
            f'{self.name.removeprefix("Monitor.")}.down',
            f'{self.name.removeprefix("Monitor.")}.ratio'
        ]

    def _param_range(self) -> dict[str, list[...]]:
        param_range = super()._param_range()

        # param_range.update(
        #     center_mode=['absolute', 'median', 'mean', 'weighted']
        # )

        return param_range

    def _param_static(self) -> dict[str, ...]:
        param_static = super()._param_static()

        param_static.update(
            center_mode='median'
        )

        param_static.update(
            weights=self.weights
        )

        return param_static

    @property
    def value(self) -> dict[str, float]:
        up_dispersion = self.dispersion(side=1, center_mode=self.center_mode)
        down_dispersion = self.dispersion(side=-1, center_mode=self.center_mode)

        if up_dispersion < 0:
            ratio = 1.
        elif down_dispersion < 0:
            ratio = 0.
        else:
            ratio = down_dispersion / (up_dispersion + down_dispersion)

        return {'up': up_dispersion, 'down': down_dispersion, 'ratio': ratio - 0.5}


class CoherenceAdaptiveMonitor(CoherenceMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = True, weights: dict[str, float] = None, center_mode='median', name: str = 'Monitor.Coherence.Price.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            weights=weights,
            center_mode=center_mode,
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

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> CoherenceAdaptiveMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            weights=json_dict['weights'],
            center_mode=json_dict['center_mode'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        return self

    def clear(self) -> None:
        AdaptiveVolumeIntervalSampler.clear(self)

        super().clear()

    def _param_static(self) -> dict[str, ...]:
        param_static = super()._param_static()

        param_static.update(
            aligned_interval=self.aligned_interval
        )

        return param_static

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class CoherenceEMAMonitor(CoherenceMonitor, EMA):
    """
    Monitors the Exponential Moving Average (EMA) of the coherence monitor.

    Inherits from CoherenceMonitor and EMA classes.
    """

    def __init__(self, sampling_interval: float, sample_size: int, alpha: float, weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Price.EMA', monitor_id: str = None):
        super().__init__(sampling_interval=sampling_interval, sample_size=sample_size, weights=weights, name=name, monitor_id=monitor_id)
        EMA.__init__(self=self, alpha=alpha)

        self.dispersion_ratio = self.register_ema(name='dispersion_ratio')

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'dispersion_ratio':
            return

        up_dispersion = self.dispersion(side=1)
        down_dispersion = self.dispersion(side=-1)

        if up_dispersion < 0:
            dispersion_ratio = 1.
        elif down_dispersion < 0:
            dispersion_ratio = 0.
        else:
            dispersion_ratio = down_dispersion / (up_dispersion + down_dispersion)

        self.update_ema(ticker='dispersion_ratio', dispersion_ratio=dispersion_ratio - 0.5)

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> CoherenceEMAMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            alpha=json_dict['alpha'],
            weights=json_dict['weights'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )
        self.update_from_json(json_dict=json_dict)

        self.dispersion_ratio = self.ema['dispersion_ratio']

        return self

    def clear(self):
        EMA.clear(self)

        super().clear()

        self.dispersion_ratio = self.register_ema(name='dispersion_ratio')

    @property
    def value(self) -> dict[str, float]:
        up_dispersion = self.dispersion(side=1)
        down_dispersion = self.dispersion(side=-1)
        return {'up': up_dispersion, 'down': down_dispersion, 'ratio': self.dispersion_ratio.get('dispersion_ratio', np.nan)}


class TradeCoherenceMonitor(CoherenceMonitor):

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float] = None, name: str = 'Monitor.Coherence.Volume', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            weights=weights,
            name=name,
            monitor_id=monitor_id,
        )

        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='volume_net', mode='accumulate')

    def __call__(self, market_data: MarketData, **kwargs):
        super().__call__(market_data=market_data)

        if isinstance(market_data, (TradeData, TransactionData)):
            self._on_trade(trade_data=market_data)

    def _on_trade(self, trade_data: TradeData | TransactionData):
        ticker = trade_data.ticker
        volume = trade_data.volume
        side = trade_data.side.sign
        timestamp = trade_data.timestamp

        self.log_obs(ticker=ticker, timestamp=timestamp, volume=volume, volume_net=volume * side)

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> TradeCoherenceMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            sample_size=json_dict['sample_size'],
            weights=json_dict['weights'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        return self

    def clear(self):
        super().clear()

        self.register_sampler(name='volume', mode='accumulate')
        self.register_sampler(name='volume_net', mode='accumulate')

    def trade_coherence(self, side: int, center_mode: str = 'median'):
        historical_price = self.get_sampler(name='price')
        volume = self.get_sampler(name='volume')
        volume_net = self.get_sampler(name='volume_net')

        valid_ticker = set(historical_price) | set(volume) | set(volume_net)

        price_pct_matrix = []
        flow_matrix = []
        weights = []

        if _ := [min(len(historical_price[ticker]), len(volume[ticker]), len(volume_net[ticker])) for ticker in valid_ticker if not (self.weights and ticker not in self.weights)]:
            vector_length = min(_)
        else:
            return np.nan

        if vector_length < 3:
            return np.nan

        for ticker in valid_ticker:
            if self.weights is not None and ticker not in self.weights:
                continue

            price_vector = np.array(list(historical_price[ticker])[-vector_length:])
            volume_vector = np.array(list(volume[ticker])[-vector_length:])
            volume_net_vector = np.array(list(volume_net[ticker])[-vector_length:])

            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            flow_vector = volume_net_vector / volume_vector
            price_pct_matrix.append(price_pct_vector)
            flow_matrix.append(flow_vector)

            weights.append(self.weights[ticker] if self.weights else 1)

        # to data snapshot
        values = []
        for price_pct_distribution, flow_distribution in zip(np.array(price_pct_matrix).T, np.array(flow_matrix).T):
            # x = rankdata(flow_distribution)
            # y = np.log(price_pct_distribution)
            if center_mode == 'absolute':
                center = 0.
            elif center_mode == 'median':
                center = np.median(price_pct_distribution)
            elif center_mode == 'mean':
                center = np.mean(price_pct_distribution)
            elif center_mode == 'weighted':
                center = np.average(price_pct_distribution, weights=weights)
            else:
                raise NotImplementedError(f'Invalid center mode {center_mode}. Expect absolute, median, mean or weighted.')

            x_selected = []
            y_selected = []
            weights_selected = []

            for weight, price_change, flow in zip(weights, price_pct_distribution, flow_distribution):
                if (price_change - center) * side <= 0:  # for entries with value == center, they will be excluded from both side
                    continue

                x_selected.append(flow * side)
                y_selected.append(abs(price_change - center))
                weights_selected.append(weight)

            x = rankdata(x_selected)
            y = np.log(y_selected)

            if len(y) < 3:
                continue

            x = np.vstack([x, np.ones(len(x))]).T

            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(X=x, y=y, sample_weight=weights_selected)
            # regressor.fit(X=x, y=y)
            slope = regressor.coef_[0]
            # from sklearn.metrics import r2_score
            # y_pred = regressor.predict(x)
            # r2 = r2_score(y_true=y, y_pred=y_pred, sample_weight=weights_selected)
            values.append(slope)

        return np.nanmean(values)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.up',
            f'{self.name.removeprefix("Monitor.")}.down'
        ]

    @property
    def value(self) -> dict[str, float]:
        up_coherence = self.trade_coherence(side=1, center_mode='median')
        down_coherence = self.trade_coherence(side=-1, center_mode='median')

        return {'up': up_coherence, 'down': down_coherence}
