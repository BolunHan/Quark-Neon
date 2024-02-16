from __future__ import annotations

import json
from collections import deque

from PyQuantKit import MarketData, TradeData, TransactionData

from .. import Synthetic, MACD, FixedIntervalSampler, AdaptiveVolumeIntervalSampler, FactorMonitor


class DivergenceMonitor(FactorMonitor, FixedIntervalSampler):
    """
    a MACD monitor to measure the divergence of the MA
    """

    def __init__(self, sampling_interval: float, name: str = 'Monitor.EMA.Divergence', monitor_id: str = None):
        super().__init__(name=name, monitor_id=monitor_id)
        FixedIntervalSampler.__init__(self=self, sampling_interval=sampling_interval, sample_size=3)

        self._macd: dict[str, MACD] = {}
        self._macd_last: dict[str, float] = {}
        self._macd_diff: dict[str, float] = {}
        self._macd_diff_ema: dict[str, float] = {}

        self.register_sampler(name='price')

        self._is_ready = True

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self.log_obs(ticker=ticker, timestamp=timestamp, price=market_price)

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'price':
            return

        price_dict = self.get_sampler(name='price')[ticker]
        idx, last_price = list(price_dict.items())[-2 if len(price_dict) > 1 else -1]  # last observation price, not the active one

        if ticker in self._macd:
            macd = self._macd[ticker]
        else:
            macd = self._macd[ticker] = MACD()

        macd.update_macd(price=last_price)
        macd_current = macd.macd_diff_adjusted
        macd_last = self._macd_last.get(ticker, 0.)
        macd_diff = macd_current - macd_last

        self._macd_last[ticker] = macd_current
        self._macd_diff[ticker] = macd_diff
        self._macd_diff_ema[ticker] = macd.update_ema(value=macd_diff, memory=self._macd_diff_ema.get(ticker, 0.), window=macd.signal_window)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            macd={ticker: macd.to_json(fmt='dict') for ticker, macd in self._macd.items()},
            macd_last=self._macd_last,
            macd_diff=self._macd_diff,
            macd_diff_ema=self._macd_diff_ema,
            is_ready=self._is_ready
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> DivergenceMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._macd = {ticker: MACD.from_json(macd_json) for ticker, macd_json in json_dict['macd'].items()}
        self._macd_last = json_dict['macd_last']
        self._macd_diff = json_dict['macd_diff']
        self._macd_diff_ema = json_dict['macd_diff_ema']
        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        FixedIntervalSampler.clear(self)

        self._macd.clear()
        self._macd_last.clear()
        self._macd_diff.clear()
        self._macd_diff_ema.clear()

        self.register_sampler(name='price')

    def macd_value(self):
        macd_dict = {}
        price_dict = self.active_obs(name='price')

        for ticker, macd in self._macd.items():
            last_price = price_dict[ticker]
            current_macd = macd.calculate_macd(price=last_price)
            macd_dict[ticker] = current_macd['macd_diff'] / last_price

        return macd_dict

    def macd_diff(self, macd_value: dict[str, float] = None):
        macd_dict = self.macd_value() if macd_value is None else macd_value

        macd_diff_dict = {}
        last_macd_dict = self._macd_last

        for ticker, macd in macd_dict.items():
            last_macd = last_macd_dict.get(ticker, 0.)
            macd_diff = macd - last_macd
            macd_diff_dict[ticker] = macd_diff

        return macd_diff_dict

    def macd_diff_ema(self, macd_diff: dict[str, float]) -> dict[str, float]:
        macd_diff_dict = self.macd_diff() if macd_diff is None else macd_diff

        macd_diff_ema_dict = {}

        for ticker, diff in macd_diff_dict.items():
            macd = self._macd[ticker]
            macd_diff_ema_dict[ticker] = macd.update_ema(value=diff, memory=self._macd_diff_ema.get(ticker, 0.), window=macd.signal_window)

        return macd_diff_ema_dict

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    def _param_static(self) -> dict[str, ...]:
        params_static = super()._param_static()

        params_static.update(
            sample_size=self.sample_size
        )

        return params_static

    @property
    def value(self) -> dict[str, float]:
        result = {}
        result.update(self.macd_value())
        return result

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class DivergenceAdaptiveMonitor(DivergenceMonitor, AdaptiveVolumeIntervalSampler):

    def __init__(self, sampling_interval: float, baseline_window: int = 15, aligned_interval: bool = False, name: str = 'Monitor.EMA.Divergence.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            name=name,
            monitor_id=monitor_id
        )

        AdaptiveVolumeIntervalSampler.__init__(
            self=self,
            sampling_interval=sampling_interval,
            sample_size=3,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval
        )

    def __call__(self, market_data: MarketData, **kwargs):
        self.accumulate_volume(market_data=market_data)
        super().__call__(market_data=market_data, **kwargs)

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> DivergenceAdaptiveMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._macd = {ticker: MACD.from_json(macd_json) for ticker, macd_json in json_dict['macd'].items()}
        self._macd_last = json_dict['macd_last']
        self._macd_diff = json_dict['macd_diff']
        self._macd_diff_ema = json_dict['macd_diff_ema']
        self._is_ready = json_dict['is_ready']

        return self

    def clear(self) -> None:
        AdaptiveVolumeIntervalSampler.clear(self)

        super().clear()

    @property
    def is_ready(self) -> bool:
        for ticker in self._volume_baseline['obs_vol_acc']:
            if ticker not in self._volume_baseline['sampling_interval']:
                return False

        return self._is_ready


class DivergenceIndexAdaptiveMonitor(DivergenceAdaptiveMonitor, Synthetic):

    def __init__(self, sampling_interval: float, baseline_window: int = 15, aligned_interval: bool = True, weights: dict[str, float] = None, name: str = 'Monitor.EMA.Divergence.Index.Adaptive', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> DivergenceIndexAdaptiveMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            weights=json_dict['weights'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._macd = {ticker: MACD.from_json(macd_json) for ticker, macd_json in json_dict['macd'].items()}
        self._macd_last = json_dict['macd_last']
        self._macd_diff = json_dict['macd_diff']
        self._macd_diff_ema = json_dict['macd_diff_ema']
        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        Synthetic.clear(self)

        super().clear()

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Index',
            f'{self.name.removeprefix("Monitor.")}.Diff',
            f'{self.name.removeprefix("Monitor.")}.Diff.EMA'
        ]

    @property
    def value(self) -> dict[str, float]:
        macd_value = self.macd_value()
        macd_diff = self.macd_diff(macd_value=macd_value)
        macd_diff_ema = self.macd_diff_ema(macd_diff=macd_diff)
        result = {'Index': self.composite(macd_value), 'Diff': self.composite(macd_diff), 'Diff.EMA': self.composite(macd_diff_ema)}
        # result.update(macd_value)

        return result


class DivergenceAdaptiveTriggerMonitor(DivergenceAdaptiveMonitor):

    def __init__(self, sampling_interval: float, confirmation_threshold: float = 0.0001, observation_window: int = 5, baseline_window: int = 15, aligned_interval: bool = False, name: str = 'Monitor.EMA.Divergence.Trigger', monitor_id: str = None):
        super().__init__(sampling_interval=sampling_interval, baseline_window=baseline_window, aligned_interval=aligned_interval, name=name, monitor_id=monitor_id)

        self.observation_window = observation_window
        self.confirmation_threshold = confirmation_threshold

        self._macd_last_extreme: dict[str, dict[str, float]] = {}
        self._macd_storage: dict[str, deque[float]] = {}

    def on_entry_added(self, ticker: str, name: str, value):
        super().on_entry_added(ticker=ticker, name=name, value=value)

        if name != 'price':
            return

        price_dict = self.get_sampler(name='price')[ticker]
        idx, last_price = list(price_dict.items())[-2 if len(price_dict) > 1 else -1]  # last observation price, not the active one

        if ticker in self._macd:
            macd = self._macd[ticker]
            macd_storage = self._macd_storage[ticker]
            macd_extreme = self._macd_last_extreme[ticker]
        else:
            macd = self._macd[ticker] = MACD()
            macd_storage = self._macd_storage[ticker] = deque(maxlen=self.observation_window)
            macd_extreme = self._macd_last_extreme[ticker] = {}

        macd.update_macd(price=last_price)
        macd_current = macd.macd_diff_adjusted
        macd_last = macd_storage[-1] if macd_storage else 0.
        macd_diff = macd_current - macd_last

        if macd_last <= 0 < macd_current:
            macd_extreme['max'] = macd_current
        elif macd_last >= 0 > macd_current:
            macd_extreme['min'] = macd_current
        elif macd_current > 0:
            macd_extreme['max'] = max(macd_current, macd_extreme.get('max', 0.))
        elif macd_current < 0:
            macd_extreme['min'] = min(macd_current, macd_extreme.get('min', 0.))
        else:
            macd_extreme['min'] = 0.
            macd_extreme['max'] = 0.

        macd_storage.append(macd_current)
        self._macd_diff[ticker] = macd_diff
        self._macd_diff_ema[ticker] = macd.update_ema(value=macd_diff, memory=self._macd_diff_ema.get(ticker, 0.), window=macd.signal_window)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict')
        data_dict.update(
            observation_window=self.observation_window,
            confirmation_threshold=self.confirmation_threshold,
            macd_last_extreme=self._macd_last_extreme,
            macd_storage={ticker: list(dq) for ticker, dq in self._macd_storage.items()},
            is_ready=self._is_ready
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> DivergenceAdaptiveTriggerMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            confirmation_threshold=json_dict['confirmation_threshold'],
            observation_window=json_dict['observation_window'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._macd = {ticker: MACD.from_json(macd_json) for ticker, macd_json in json_dict['macd'].items()}
        self._macd_last = json_dict['macd_last']
        self._macd_diff = json_dict['macd_diff']
        self._macd_diff_ema = json_dict['macd_diff_ema']

        self._macd_last_extreme = json_dict['macd_last_extreme']
        self._macd_storage = {ticker: deque(storage, maxlen=self.observation_window) for ticker, storage in json_dict['macd_storage'].items()}

        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        super().clear()
        self._macd_last_extreme.clear()

    def _param_range(self) -> dict[str, list[...]]:
        param_range = super()._param_range()

        param_range.update(
            confirmation_threshold=[0.0001, 0.0002, 0.0005],
            observation_window=[5, 10, 15]
        )

        return param_range

    @property
    def value(self) -> dict[str, float]:
        monitor_value = {}

        for ticker, storage in self._macd_storage.items():
            last_extreme = self._macd_last_extreme[ticker]

            if storage:
                if (
                        min(storage) <= 0. and
                        last_extreme.get('min', 0.) <= -self.confirmation_threshold and
                        storage[-1] > 0
                ):
                    state = 1.
                elif (
                        max(storage) >= 0. and
                        last_extreme.get('max', 0.) >= self.confirmation_threshold and
                        storage[-1] < 0
                ):
                    state = -1.
                else:
                    state = 0.

                monitor_value[ticker] = state

        return monitor_value

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class DivergenceAdaptiveTriggerIndexMonitor(DivergenceAdaptiveTriggerMonitor, Synthetic):
    def __init__(self, sampling_interval: float, confirmation_threshold: float = 0.0001, observation_window: int = 5, baseline_window: int = 15, aligned_interval: bool = False, weights: dict[str, float] = None, name: str = 'Monitor.EMA.Divergence.Trigger', monitor_id: str = None):
        super().__init__(
            sampling_interval=sampling_interval,
            confirmation_threshold=confirmation_threshold,
            observation_window=observation_window,
            baseline_window=baseline_window,
            aligned_interval=aligned_interval,
            name=name,
            monitor_id=monitor_id
        )

        Synthetic.__init__(self=self, weights=weights)

    def __call__(self, market_data: MarketData, **kwargs):
        super().__call__(market_data=market_data, **kwargs)
        self.update_synthetic(ticker=market_data.ticker, market_price=market_data.market_price)
        self.accumulate_volume(ticker='Synthetic', volume=market_data.notional if isinstance(market_data, (TradeData, TransactionData)) else 0)
        self.log_obs(ticker='Synthetic', timestamp=market_data.timestamp, price=self.synthetic_index)

    def factor_names(self, subscription: list[str]) -> list[str]:
        return [
            f'{self.name.removeprefix("Monitor.")}.Synthetic',
        ] + [
            f'{self.name.removeprefix("Monitor.")}.{ticker}' for ticker in subscription
        ]

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> DivergenceAdaptiveTriggerIndexMonitor:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            sampling_interval=json_dict['sampling_interval'],
            confirmation_threshold=json_dict['confirmation_threshold'],
            observation_window=json_dict['observation_window'],
            baseline_window=json_dict['baseline_window'],
            aligned_interval=json_dict['aligned_interval'],
            weights=json_dict['weights'],
            name=json_dict['name'],
            monitor_id=json_dict['monitor_id']
        )

        self.update_from_json(json_dict=json_dict)

        self._macd = {ticker: MACD.from_json(macd_json) for ticker, macd_json in json_dict['macd'].items()}
        self._macd_last = json_dict['macd_last']
        self._macd_diff = json_dict['macd_diff']
        self._macd_diff_ema = json_dict['macd_diff_ema']

        self._macd_last_extreme = json_dict['macd_last_extreme']
        self._macd_storage = {ticker: deque(storage, maxlen=self.observation_window) for ticker, storage in json_dict['macd_storage'].items()}

        self._is_ready = json_dict['is_ready']

        return self

    def clear(self):
        Synthetic.clear(self)

        super().clear()
