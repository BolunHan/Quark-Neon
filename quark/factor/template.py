__package__ = 'quark.factor'

__all__ = ['SamplerMonitor']

__meta__ = {
    'ver': '1.0.0',
    'name': 'SamplerMonitor',
    'params': [
        dict(
            sampling_interval=15,
            name='Monitor.SamplerMonitor15'
        ),
        dict(
            sampling_interval=30,
            name='Monitor.SamplerMonitor.30'
        ),
        dict(
            sampling_interval=60,
            name='Monitor.SamplerMonitor.60'
        )
    ],
    'family': 'template',
    'market': 'cn',
    'requirements': [
        'quark @ git+https://github.com/BolunHan/Quark.git#egg=Quark',  # this requirement is not necessary, only put there as a demo
        'numpy',  # not necessary too, since the env installed with numpy by default
        'PyAlgoEngine',  # also not necessary
    ],
    'external_data': [],  # topic for required external data
    'external_lib': [],  # exotic libraries, like compiled c lib file.
    'factor_type': 'basic',
    'activated_date': None,
    'deactivated_date': None,
    'dependencies': [],
    'author': 'Bolun',
    'comments':
        "A basic template for factor monitor building."
}

from collections import defaultdict, deque

import numpy as np
from algo_engine.base import TradeData, TransactionData

from .utils import FactorMonitor, Synthetic, IndexWeight
from .sampler import FixedIntervalSampler, SamplerMode, SamplerData


class SamplerMonitor(FactorMonitor, FixedIntervalSampler, Synthetic):
    def __init__(self, sampling_interval: int, weights: IndexWeight, name: str = 'Monitor.Sampler.Fixed'):
        super().__init__(name=name, filter_mode=7)
        FixedIntervalSampler.__init__(
            self,
            sampling_interval=sampling_interval,
            sample_size=int(3600 / sampling_interval),
            # baseline_window=int(1800 / sampling_interval),
            # profile_type=VolumeProfileType.interval_volume,
            # interval=5 * 60,
            # profile=GlobalStatics.PROFILE
        )
        Synthetic.__init__(self, weights=weights)

        self.traceback = 10
        self.register_sampler(topic='price', mode=SamplerMode.update)
        self.vwap = {}
        self.n_call = defaultdict(lambda: deque(maxlen=30))

    def on_triggered(self, ticker: str, topic: str, sampler: SamplerData, **kwargs):
        price_history = sampler.history[ticker]
        self.vwap[ticker] = np.mean(price_history)
        n_call = self.n_call[ticker]
        if n_call:
            n_call[-1] += 1
        else:
            n_call.append(1)

    def on_trade_data(self, trade_data: TradeData | TransactionData, **kwargs) -> None:
        # self.accumulate_volume(market_data=trade_data)
        self.log_obs(ticker=trade_data.ticker, timestamp=trade_data.timestamp, price=trade_data.price)
        # pass

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.vwap',
            f'{self.name.removeprefix("Monitor.")}.n_call'
        ]

    @property
    def value(self) -> dict[str, float] | float:
        _ = {
            'vwap': self.composite(self.vwap, replace_na=np.nan),
            'n_call': sum([np.mean(list(dq)) for dq in self.n_call.values()])
        }

        for dq in self.n_call.values():
            dq.append(0)

        return _
