from collections import deque
from typing import Iterable

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData

from .. import Synthetic, FactorMonitor, FixedIntervalSampler, AdaptiveVolumeIntervalSampler


class TradeFlowMonitor(FactorMonitor, FixedIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int, name: str = 'Monitor.TradeFlow', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.register_sampler(name='price', mode='update')
        self.register_sampler(name='trade_flow', mode='accumulate')
        self.register_sampler(name='volume', mode='accumulate')

        self._historical_trade_imbalance = {}
        self._is_ready = True

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        ticker = trade_data.ticker
        timestamp = trade_data.timestamp
        market_price = trade_data.market_price
        volume = trade_data.volume
        side = trade_data.side.sign

        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, volume=volume, trade_flow=volume * side)

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'trade_flow':
            return

        trade_imbalance_dict = self.trade_imbalance(ticker=ticker, drop_last=True, boosted=False)

        if ticker not in trade_imbalance_dict:
            return

        trade_imbalance = trade_imbalance_dict[ticker]

        if not np.isfinite(trade_imbalance):
            return

        if ticker in self._historical_trade_imbalance:
            historical_trade_imbalance = self._historical_trade_imbalance[ticker]
        else:
            historical_trade_imbalance = self._historical_trade_imbalance[ticker] = deque(maxlen=max(5, int(self.sample_size / 2)))

        historical_trade_imbalance.append(trade_imbalance)

    def slope(self) -> dict[str, float]:
        slope_dict = {}
        for ticker in self._historical_trade_imbalance:
            trade_imbalance = list(self._historical_trade_imbalance[ticker])

            if len(trade_imbalance) < 3:
                slope = np.nan
            else:
                x = list(range(len(trade_imbalance)))
                x = np.vstack([x, np.ones(len(x))]).T
                y = np.array(trade_imbalance)

                slope, c = np.linalg.lstsq(x, y, rcond=None)[0]

            slope_dict[ticker] = slope

        return slope_dict

    def clear(self):
        FixedIntervalSampler.clear(self)

        self._historical_trade_imbalance.clear()

        self.register_sampler(name='price', mode='update')
        self.register_sampler(name='trade_flow', mode='accumulate')
        self.register_sampler(name='volume', mode='accumulate')

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    @classmethod
    def sampling(cls, observation: np.ndarray, n_bins: int):
        """
        Sample continuous variables X1 and X2 into discrete bins based on quantiles.

        Parameters:
        - observation: 2D array representing observations of X1 and X2.
        - n_bins: Number of bins for discretization.

        Returns:
        - joint_distribution: 2D array representing the joint distribution of variables X1 and X2.
        """
        # Discretize X1 and X2 into bins
        x1_bins = np.linspace(np.min(observation[:, 0]), np.max(observation[:, 0]), n_bins + 1)
        x2_bins = np.linspace(np.min(observation[:, 1]), np.max(observation[:, 1]), n_bins + 1)

        # Compute indices of the bins for each observation
        x1_indices = np.digitize(observation[:, 0], x1_bins)
        x2_indices = np.digitize(observation[:, 1], x2_bins)

        # Count observations falling into each bin
        joint_distribution, _, _ = np.histogram2d(x1_indices, x2_indices, bins=(range(1, len(x1_bins) + 1), range(1, len(x2_bins) + 1)))

        # Normalize joint distribution
        joint_distribution /= np.sum(joint_distribution)

        return joint_distribution

    @classmethod
    def mutual_information(cls, joint_distribution: np.ndarray, epsilon: float = 1e-10):
        """
        Compute the mutual information of two variables given their joint distribution.

        Parameters:
        - joint_distribution: 2D array representing the joint distribution of variables X1 and X2.

        Returns:
        - Mutual information (float).
        """
        # Compute marginal distributions
        marginal_x1 = np.sum(joint_distribution, axis=1)
        marginal_x2 = np.sum(joint_distribution, axis=0)

        # Compute individual entropies
        entropy_x1 = -np.sum(marginal_x1 * np.log(marginal_x1 + epsilon))  # Adding a small epsilon to avoid log(0)
        entropy_x2 = -np.sum(marginal_x2 * np.log(marginal_x2 + epsilon))
        entropy_joint = -np.sum(joint_distribution * np.log(joint_distribution + epsilon))

        # Compute mutual information
        mutual_information = entropy_x1 + entropy_x2 - entropy_joint

        return mutual_information

    def trade_imbalance_info(self):
        """
        I(trade_imbalance; price_pct) = H(x1) + H(x2) - H(x1, x2)
        Returns:
        """
        volume_dict = self.get_sampler(name='volume')
        trade_flow_dict = self.get_sampler(name='trade_flow')
        price_dict = self.get_sampler(name='price')
        result = {}

        for ticker in volume_dict:
            price_vector = np.array(list(price_dict[ticker].values()))
            price_pct_vector = np.diff(price_vector) / price_vector[:-1]
            volume_vector = np.array(list(volume_dict[ticker].values()))[1:]
            trade_flow_vector = np.array(list(trade_flow_dict[ticker].values()))[1:]
            trade_imbalance = np.array([trade_flow / volume if volume else 0. for trade_flow, volume in zip(trade_flow_vector, volume_vector)])
            observation = np.array([price_pct_vector, trade_imbalance]).T

            u_u = np.sum([1 for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct > 0 and trade_imbalance > 0])
            u_d = np.sum([1 for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct > 0 and trade_imbalance < 0])
            d_u = np.sum([1 for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct < 0 and trade_imbalance > 0])
            d_d = np.sum([1 for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct < 0 and trade_imbalance < 0])
            n = u_u + u_d + d_u + d_d

            if not n:
                continue

            joint_distribution = np.array([
                [u_u / n, u_d / n],
                [d_u / n, d_d / n]
            ])

            # joint_distribution = self.sampling(observation=observation, n_bins=4)
            # mutual_information = self.mutual_information(joint_distribution=joint_distribution)
            joint_info = -np.sum(joint_distribution * np.log(joint_distribution + 1e-5))

            result[ticker] = joint_info

        return result

    def trade_imbalance(self, ticker: str = None, drop_last: bool = False, boosted: bool = True):
        volume_dict = self.get_sampler(name='volume')
        trade_flow_dict = self.get_sampler(name='trade_flow')
        price_dict = self.get_sampler(name='price')
        result = {}

        if ticker is None:
            tasks = list(volume_dict)
        elif isinstance(ticker, str):
            tasks = [ticker]
        elif isinstance(ticker, Iterable):
            tasks = list(ticker)
        else:
            raise TypeError(f'Invalid ticker {ticker}, expect str, list[str] or None.')

        for ticker in tasks:
            price_vector = list(price_dict[ticker].values())
            volume_vector = list(volume_dict[ticker].values())[1:]
            trade_flow_vector = list(trade_flow_dict[ticker].values())[1:]

            if len(price_vector) < 4:
                continue

            if drop_last:
                price_vector.pop(-1)
                volume_vector.pop(-1)
                trade_flow_vector.pop(-1)

            price_pct_vector = np.diff(price_vector) / np.array(price_vector)[:-1]
            trade_imbalance = np.array([trade_flow / volume if volume else 0. for trade_flow, volume in zip(trade_flow_vector, volume_vector)])

            if boosted:
                u_u = np.sum([trade_imbalance * price_pct for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct > 0 and trade_imbalance > 0])
                d_d = np.sum([trade_imbalance * price_pct for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct < 0 and trade_imbalance < 0])

                result[ticker] = u_u - d_d
            else:
                u_u = np.sum([trade_imbalance for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct > 0 and trade_imbalance > 0])
                u_d = np.sum([trade_imbalance for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct > 0 and trade_imbalance < 0])
                d_u = np.sum([trade_imbalance for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct < 0 and trade_imbalance > 0])
                d_d = np.sum([trade_imbalance for price_pct, trade_imbalance in zip(price_pct_vector, trade_imbalance) if price_pct < 0 and trade_imbalance < 0])

                result[ticker] = u_u + d_d + u_d + d_u
                result[ticker] = u_u + d_d

        return result

    @property
    def value(self) -> dict[str, float]:
        return self.trade_imbalance(boosted=True)

    @property
    def is_ready(self) -> bool:
        """
        Check if the monitor is ready.

        Returns:
        bool: True if the monitor is ready, False otherwise.
        """
        return self._is_ready


class TradeFlowAdaptiveMonitor(TradeFlowMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, sample_size: int = 20, baseline_window: int = 100, aligned_interval: bool = False, name: str = 'Monitor.TradeFlow.Adaptive', monitor_id: str = None):
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
        for ticker in self._volume_baseline['obs_vol_acc']:
            if ticker not in self._volume_baseline['sampling_interval']:
                return False

        return self._is_ready


class TradeFlowAdaptiveIndexMonitor(TradeFlowAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, sample_size: int, baseline_window: int, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.TradeFlow.Adaptive.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker

        if self.weights and ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def clear(self) -> None:
        Synthetic.clear(self)

        super().clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Imbalance',
            f'{self.name.removeprefix("Monitor.")}.Entropy',
            f'{self.name.removeprefix("Monitor.")}.Boosted',
            f'{self.name.removeprefix("Monitor.")}.Slope',
        ]

    @property
    def value(self) -> dict[str, float]:
        trade_imbalance = self.trade_imbalance(boosted=False)
        trade_imbalance_info = self.trade_imbalance_info()
        trade_imbalance_boosted = self.trade_imbalance(boosted=True)
        slope = self.slope()

        return {
            'Imbalance': self.composite(values=trade_imbalance),
            'Entropy': self.composite(values=trade_imbalance_info),
            'Boosted': self.composite(values=trade_imbalance_boosted),
            'Slope': self.composite(values=slope)
        }
