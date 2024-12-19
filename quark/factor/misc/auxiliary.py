from algo_engine.base import MarketData, TradeData, TransactionData

from .. import SamplerMode, Synthetic, FactorMonitor, VolumeProfileSampler, VolumeProfileType, AdaptiveVolumeIntervalSampler, FixedIntervalSampler, ALPHA_001, EMA, SamplerData

__all__ = ['TrendAuxiliaryMonitor', 'TrendAdaptiveAuxiliaryMonitor', 'TrendVolumeProfileAuxiliaryMonitor', 'TrendIndexAdaptiveAuxiliaryMonitor', 'TrendIndexVolumeProfileAuxiliaryMonitor']


class TrendAuxiliaryMonitor(FactorMonitor, FixedIntervalSampler):
    def __init__(self, sampling_interval: float, sample_size: int, alpha: float = 1 - ALPHA_001, name: str = 'Monitor.Trend', monitor_id: str = None):
        self.alpha = alpha

        super().__init__(name=name, monitor_id=monitor_id, filter_mode=0x07)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=sample_size)
        self.ema = EMA(alpha=self.alpha)

        self.register_sampler(topic='price', mode=SamplerMode.update)
        self.register_sampler(topic='notional', mode=SamplerMode.accumulate)
        self.ema.register_ema(name='pct_change')

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        ticker = trade_data.ticker
        timestamp = trade_data.timestamp
        price = trade_data.price

        self.log_obs(ticker=ticker, timestamp=timestamp, price=price)

    def on_triggered(self, ticker: str, topic: str, sampler: SamplerData, **kwargs):
        price_history = sampler.history[ticker]

        if len(price_history) >= 2:
            latest_pct_change = (price_history[-1] / price_history[-2]) - 1
            self.ema.update_ema(ticker=ticker, pct_change=latest_pct_change)
            self.ema.enroll(ticker=ticker, name='pct_change')

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    @property
    def value(self) -> dict[str, float]:
        return self.ema.ema.get('pct_change', {})


class TrendAdaptiveAuxiliaryMonitor(TrendAuxiliaryMonitor, AdaptiveVolumeIntervalSampler):
    def __init__(self, sampling_interval: float, sample_size: int, alpha: float = ALPHA_001, baseline_window: int = 100, aligned_interval: bool = False, name: str = 'Monitor.Trend.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            alpha=alpha,
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

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        self.accumulate_volume(market_data=trade_data)
        super().on_trade_data(trade_data=trade_data, **kwargs)

    @property
    def is_ready(self) -> bool:
        return self.baseline_ready


class TrendVolumeProfileAuxiliaryMonitor(TrendAuxiliaryMonitor, VolumeProfileSampler):
    def __init__(self, sampling_interval: float, sample_size: int, alpha: float = ALPHA_001, name: str = 'Monitor.Trend.VolumeProfile', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            alpha=alpha,
            name=name,
            monitor_id=monitor_id
        )

        VolumeProfileSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            profile_type=VolumeProfileType.interval_volume,
        )

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs):
        self.accumulate_volume(market_data=trade_data)
        super().on_trade_data(trade_data=trade_data, **kwargs)

    @property
    def is_ready(self) -> bool:
        return self.profile_ready


class TrendIndexAdaptiveAuxiliaryMonitor(TrendAdaptiveAuxiliaryMonitor, Synthetic):
    def __init__(self, sampling_interval: float, sample_size: int, alpha: float = ALPHA_001, baseline_window: int = 100, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.Trend.Adaptive.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            alpha=alpha,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        if self.weights and market_data.ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Trend',
        ]

    @property
    def value(self) -> dict[str, float]:
        return {
            'Trend': self.composite(values=self.ema.ema.get('pct_change', {})),
        }


class TrendIndexVolumeProfileAuxiliaryMonitor(TrendVolumeProfileAuxiliaryMonitor, Synthetic):
    def __init__(self, sampling_interval: float, sample_size: int, alpha: float = ALPHA_001, weights: dict[str, float] = None, name: str = 'Monitor.Trend.VolumeProfile.Index', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            sample_size=sample_size,
            alpha=alpha,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        if self.weights and market_data.ticker not in self.weights:
            return

        super().__call__(market_data=market_data, **kwargs)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Trend',
        ]

    @property
    def value(self) -> dict[str, float]:
        return {
            'Trend': self.composite(values=self.ema.ema.get('pct_change', {})),
        }
