import datetime
import json
from typing import Any

import numpy as np
from algo_engine.base import MarketData, TradeData, TransactionData
from quark.factor import FactorMonitor, FixedIntervalSampler, VolumeProfileSampler, SamplerMode, VolumeProfileType, SamplerData


class EntropyMonitor(FactorMonitor, FixedIntervalSampler):
    """
    Monitors and measures the entropy of the covariance matrix.

    The entropy measures the information coming from two parts:
    - The variance of the series.
    - The inter-connection of the series.

    If we ignore the primary trend, which is mostly the standard deviation (std),
    the entropy mainly denotes the coherence of the price vectors.

    A large entropy generally indicates the end of a trend.

    Attributes:
        sample_size (int): Max sample size.
        sampling_interval (float): Time interval for sampling market data.
        weights (dict[str, float]): Weights for individual stocks in the pool.
        ignore_primary (bool): Whether to ignore the primary component (std) of the covariance matrix.
        name (str): Name of the monitor.
        monitor_id (str): Identifier for the monitor.
    """

    def __init__(self, sampling_interval: float, sample_size: int, weights: dict[str, float], pct_change: bool = False, ignore_primary: bool = True, name: str = 'Monitor.Entropy', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id, filter_mode=0x07)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)

        self.weights = weights
        self.pct_change = pct_change
        self.ignore_primary = ignore_primary

        self.register_sampler(topic='price', mode=SamplerMode.update)
        self.register_sampler(topic='notional', mode=SamplerMode.accumulate)
        self.register_sampler(topic='notional_buy', mode=SamplerMode.accumulate)
        self.register_sampler(topic='notional_sell', mode=SamplerMode.accumulate)

        self._entropy = {}

    def __call__(self, market_data: MarketData, **kwargs):
        if self.weights and market_data.ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        ticker = trade_data.ticker

        if ticker not in self.weights:
            return

        market_price = trade_data.market_price
        timestamp = trade_data.timestamp
        notional = trade_data.notional
        side = trade_data.side.sign

        if side > 0:
            self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, notional=notional, notional_buy=notional, notional_sell=0)
        elif side < 0:
            self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price, notional=notional, notional_buy=0, notional_sell=notional)

    def on_triggered(self, ticker: str, topic: str, sampler: SamplerData, **kwargs):
        if topic != 'price':
            return

        self._entropy.clear()

    def clear(self) -> None:
        super().clear()

        self.register_sampler(topic='price', mode=SamplerMode.update)
        self.register_sampler(topic='notional', mode=SamplerMode.accumulate)
        self.register_sampler(topic='notional_buy', mode=SamplerMode.accumulate)
        self.register_sampler(topic='notional_sell', mode=SamplerMode.accumulate)

        self._entropy.clear()

    @classmethod
    def covariance_matrix(cls, vectors: list[list[float]] | np.ndarray) -> np.ndarray:
        """
        Calculates the covariance matrix of the given vectors.

        Args:
            vectors (list[list[float]]): List of vectors.

        Returns:
            np.ndarray: Covariance matrix.
        """
        data = np.array(vectors)
        matrix = np.cov(data, ddof=0, rowvar=True)
        return matrix

    @classmethod
    def entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        """
        Calculates the entropy of the given covariance matrix.

        Args:
            matrix (list[list[float]] | np.ndarray): Covariance matrix.

        Returns:
            float: Entropy value.
        """
        # Note: The matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # Just to be safe
        e = e[e > 0]

        t = e * np.log2(e)
        return -np.sum(t)

    @classmethod
    def secondary_entropy(cls, matrix: list[list[float]] | np.ndarray) -> float:
        """
        Calculates the secondary entropy of the given covariance matrix.

        Args:
            matrix (list[list[float]] | np.ndarray): Covariance matrix.

        Returns:
            float: Secondary entropy value.
        """
        # Note: The matrix is a covariance matrix, which is always positive semi-definite
        e = np.linalg.eigvalsh(matrix)
        # Just to be safe
        e = e[e > 0]

        # Remove the primary component (std) of the covariance matrix
        primary_index = np.argmax(e)
        e = np.delete(e, primary_index)

        t = e * np.log2(e)
        return -np.sum(t)

    def _param_static(self) -> dict[str, ...]:
        param_static = super()._param_static()

        param_static.update(
            weights=self.weights,
            pct_change=self.pct_change,
            ignore_primary=self.ignore_primary
        )

        return param_static

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            weights=dict(self.weights),
            pct_change=self.pct_change,
            ignore_primary=self.ignore_primary
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    def get_cov(self, topic: str) -> np.ndarray[Any, np.dtype[np.float64]]:
        history = []

        match topic:
            case 'price' | 'notional' | 'notional_buy' | 'notional_sell':
                _history = self.get_history(topic=topic)
                _latest = self.get_latest(topic=topic)
                median_length = int(np.median([len(_history[ticker]) if ticker in _history else 0 for ticker in self.weights]))

                for ticker, weight in self.weights.items():
                    if ticker not in _history:
                        # raise ValueError(f'Too few observations! {self.__class__.__name__} not ready!')
                        history.append(np.ones(shape=(median_length,)) * _latest.get(ticker, 0.))
                    else:
                        history.append(np.array(_history[ticker]) * weight)
            case 'price_pct_change':
                _history = self.get_history(topic='price')
                median_length = int(np.median([len(_history[ticker]) if ticker in _history else 0 for ticker in self.weights]))

                for ticker in self.weights:
                    if ticker not in _history:
                        # raise ValueError(f'Too few observations! {self.__class__.__name__} not ready!')
                        history.append(np.zeros(shape=(median_length,)))
                    else:
                        data = _history[ticker]
                        history.append(np.diff(data) / np.array(data)[:-1])
            case 'notional_buy_pct' | 'notional_sell_pct':
                notional_sel = self.get_history(topic=topic[:-4])
                notional_ttl = self.get_history(topic='notional')
                median_length = int(np.median([len(notional_ttl[ticker]) if ticker in notional_ttl else 0 for ticker in self.weights]))

                if median_length < self.sample_size / 2:
                    raise ValueError(f'Too few observations! {self.__class__.__name__} not ready!')

                for ticker in self.weights:
                    if ticker not in notional_sel:
                        # raise ValueError(f'Too few observations! {self.__class__.__name__} not ready!')
                        history.append(np.ones(shape=(median_length,)) * 0.5)
                    else:
                        history.append(np.nan_to_num(np.divide(np.array(notional_sel[ticker]), np.array(notional_ttl[ticker])), nan=0.5, posinf=1., neginf=0))
            case _:
                raise NotImplementedError(f'{self.__class__.__name__} {topic=} not implemented.')

        if (vector_length := min([len(_) for _ in history])) < 3:
            raise ValueError(f'Too few observations! {self.__class__.__name__} not ready!')

        matrix = np.array([_[-vector_length:] for _ in history])
        cov = self.covariance_matrix(vectors=matrix)
        return cov

    def get_entropy(self, topic: str = None, cov: np.ndarray = None, ignore_primary: bool = False) -> float:
        if cov is None:
            cov = self.get_cov(topic=topic)

        if ignore_primary:
            entropy = self.secondary_entropy(matrix=cov)
        else:
            entropy = self.entropy(matrix=cov)

        return entropy

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.price_entropy',
            f'{self.name.removeprefix("Monitor.")}.price_secondary_entropy',
            f'{self.name.removeprefix("Monitor.")}.price_pct_entropy',
            # f'{self.name.removeprefix("Monitor.")}.price_pct_secondary_entropy',
            # f'{self.name.removeprefix("Monitor.")}.notional_pct_entropy',
            # f'{self.name.removeprefix("Monitor.")}.notional_pct_secondary_entropy',
        ]

    def calculate_entropy(self, override_cache: bool = False) -> dict[str, float]:
        entropy = self._entropy

        if override_cache:
            entropy.clear()

        if 'price_entropy' not in entropy:
            try:
                price_cov = self.get_cov(topic='price')
                entropy.update(
                    price_entropy=self.get_entropy(cov=price_cov, ignore_primary=False),
                    price_secondary_entropy=self.get_entropy(cov=price_cov, ignore_primary=True)
                )
            except ValueError:
                pass

        if 'price_pct_entropy' not in entropy:
            try:
                price_pct_cov = self.get_cov(topic='price_pct_change')
                entropy.update(
                    price_pct_entropy=self.get_entropy(cov=price_pct_cov, ignore_primary=False),
                    # price_pct_secondary_entropy=self.get_entropy(cov=price_pct_cov, ignore_primary=True)
                )
            except ValueError:
                pass

        # if 'notional_pct_entropy' not in entropy:
        #     try:
        #         notional_buy_cov = self.get_cov(topic='notional_buy_pct')
        #         entropy.update(
        #             notional_pct_entropy=self.get_entropy(cov=notional_buy_cov, ignore_primary=False),
        #             # notional_pct_secondary_entropy=self.get_entropy(cov=notional_buy_cov, ignore_primary=True),
        #         )
        #     except ValueError:
        #         pass

        return entropy

    @property
    def value(self) -> dict[str, float]:
        entropy = self.calculate_entropy()
        return entropy

    @property
    def is_ready(self) -> bool:
        """
        Checks if the EntropyMonitor is ready.

        Returns:
            bool: True if the monitor is ready, False otherwise.
        """
        _history = self.get_history(topic='price')

        for ticker in self.weights:
            if ticker not in _history:
                return False

            if len(_history[ticker]) != self.sample_size:
                return False

        return True


class EntropyVolumeProfileMonitor(EntropyMonitor, VolumeProfileSampler):
    def __init__(self, sampling_interval: float, sample_size: int = 20, weights: dict[str, float] = None, pct_change: bool = False, ignore_primary: bool = True, name: str = 'Monitor.Entropy.VolumeProfile', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            weights=weights,
            pct_change=pct_change,
            ignore_primary=ignore_primary,
            name=name,
            monitor_id=monitor_id
        )

        VolumeProfileSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            profile_type=VolumeProfileType.interval_volume,
            sample_size=sample_size,
        )

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        self.accumulate_volume(market_data=trade_data)
        super().on_trade_data(trade_data=trade_data, **kwargs)

    @property
    def is_ready(self) -> bool:
        readiness = self.contexts.get('readiness_flag')
        if readiness:
            return True

        if super().is_ready and self.profile_ready:
            self.contexts['readiness_flag'] = True
            return True

        profile_class = self.contexts['profile_type'].get_profile()
        session_start = self.mds.session_start
        ts_start = 0 if session_start is None else profile_class._time_to_seconds(session_start)
        market_time = self.mds.market_time
        ts_now = 0 if market_time is None else profile_class._time_to_seconds(market_time)

        if ts_now - ts_start > self.sample_size * self.sampling_interval:
            self.contexts['readiness_flag'] = True
            return True

        return False
