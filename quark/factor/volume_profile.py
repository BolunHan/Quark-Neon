from __future__ import annotations

import abc
import copy
import ctypes
import datetime
import enum
import inspect
import json
from collections import deque
from functools import cached_property
from typing import Literal, Iterable, overload, Self

import numpy as np
from algo_engine.base import MarketData, TickData, TradeData, TransactionData, BarData
from algo_engine.profile import Profile

from .. import LOGGER
from ..base import GlobalStatics


class ProfileType(enum.StrEnum):
    simple_online = enum.auto()

    def get_profile(self) -> type[VolumeProfile]:
        match self:
            case "simple_online":
                return SimpleOnlineProfile
            case "accumulative_volume":
                return AccumulatedVolumeProfile
            case "interval_volume":
                return IntervalVolumeProfile
            case _:
                raise NotImplementedError(f'Invalid profile {self}!')

    def to_profile(self, **kwargs) -> VolumeProfile:
        return self.get_profile()(**kwargs)


class SingleObs(ctypes.Structure):
    _fields_ = [
        ('acc_vol', ctypes.c_double),
        ('acc_amt', ctypes.c_double),
        ('px_range', ctypes.c_double),
        ('px_diff', ctypes.c_double),
        # --- still need to log the original data ---
        ('ts', ctypes.c_double),
        ('px_open', ctypes.c_double),
        ('px_close', ctypes.c_double),
        ('px_high', ctypes.c_double),
        ('px_low', ctypes.c_double),
        # --- flags ---
        ('is_init', ctypes.c_bool),
    ]

    def to_list(self):
        """the list must be in the same order as the ctype array"""
        return [self.acc_vol, self.acc_amt, self.px_range, self.px_diff, self.ts, self.px_open, self.px_close, self.px_high, self.px_low, self.is_init]

    @classmethod
    def from_list(cls, data):
        return cls(*data)


class ObsArray(object):
    def __init__(self, n: int, interval: float, buffer: ctypes.Array[SingleObs] = None):
        self.n = n
        self.obs_interval = interval
        self.buffer = self.get_buffer() if buffer is None else buffer
        self.ts = 0.
        self.y = 0.

    def __iter__(self):
        return iter(self.buffer)

    @overload
    def __getitem__(self, i: int) -> SingleObs:
        ...

    @overload
    def __getitem__(self, i: slice) -> Self:
        ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            buffer_slice: ctypes.Array[SingleObs] = self.buffer[index]
            buffer_length = len(buffer_slice)
            new_buffer = ObsArray(n=buffer_length, interval=self.obs_interval, buffer=buffer_slice)
            return new_buffer
        elif isinstance(index, int):
            return self.buffer[index]
        else:
            raise IndexError("Index must be an int or slice!")

    def to_list(self):
        return [(obs.to_list()) for obs in self]

    def flatten(self):
        return np.array([_ for _obs in self.buffer for _ in (_obs.acc_amt, _obs.px_range, _obs.px_diff)])

    def update(self, market_data: MarketData, ts: float, idx: int = None):
        self.ts = ts

        if idx is None:
            idx, res = divmod(ts, self.obs_interval)

            if idx and not res and isinstance(market_data, (TickData, BarData)):
                idx -= 1

            idx = int(idx)

        if idx >= len(self.buffer):
            LOGGER.debug(f'Overflow detected at volume_profile observation, {market_data=}, {idx=}.')
            idx = len(self.buffer) - 1
            return

        obs: SingleObs = self.buffer[idx]
        market_price = market_data.market_price

        if not obs.is_init:
            match market_data:
                case BarData():
                    market_data: BarData
                    obs.px_open = market_data.open_price
                    obs.px_close = market_data.close_price
                    obs.px_high = market_data.high_price
                    obs.px_low = market_data.low_price
                case _:
                    obs.px_open = market_price
                    obs.px_close = market_price
                    obs.px_high = market_price
                    obs.px_low = market_price
            obs.is_init = True

        match market_data:
            case BarData():
                market_data: BarData
                obs.px_close = market_data.close_price
                obs.px_high = max(obs.px_high, market_data.high_price)
                obs.px_low = min(obs.px_low, market_data.low_price)
                obs.acc_vol += market_data.volume
                obs.acc_amt += market_data.notional
            case TradeData() | TransactionData():
                market_data: TradeData | TransactionData
                obs.px_close = market_data.price
                obs.px_high = max(obs.px_high, market_data.price)
                obs.px_low = min(obs.px_low, market_data.price)
                obs.acc_vol += market_data.volume
                obs.acc_amt += market_data.notional
            case TickData():
                market_data: TickData
                pre_obs = self.buffer[int(idx) - 1] if idx > 0 else None
                pre_acc_vol = pre_obs.acc_vol if pre_obs is not None else 0
                pre_acc_amt = pre_obs.acc_amt if pre_obs is not None else 0

                obs.px_close = market_data.last_price
                obs.px_high = max(obs.px_high, market_data.last_price)
                obs.px_low = min(obs.px_low, market_data.last_price)
                obs.acc_vol = market_data.total_traded_volume - pre_acc_vol
                obs.acc_amt = market_data.total_traded_notional - pre_acc_amt
            case _:
                obs.px_close = market_data.market_price
                obs.px_high = max(obs.px_high, market_data.market_price)
                obs.px_low = min(obs.px_low, market_data.market_price)

        obs.px_range = obs.px_high - obs.px_low
        obs.px_diff = obs.px_close - obs.px_open

    def get_buffer(self) -> ctypes.Array[SingleObs]:
        buffer = (self.n * SingleObs)()

        for i, obs in enumerate(buffer, start=1):
            obs.ts = i * self.obs_interval

        return buffer

    @classmethod
    def sma_baseline(cls, window: int, x: list[ObsArray]) -> list[ObsArray]:
        obs_storage = deque(maxlen=window)
        baseline = []

        for obs_array in x:
            if not obs_storage:
                baseline.append(obs_array.__copy__())
                obs_storage.append(obs_array)
                continue

            obs_storage.append(obs_array)

            baseline_buffer = cls(n=obs_array.n, interval=obs_array.obs_interval)
            for i, _baseline in enumerate(baseline_buffer):
                _baseline: SingleObs
                _baseline.acc_vol = np.mean([_x[i].acc_vol for _x in obs_storage])
                _baseline.acc_amt = np.mean([_x[i].acc_amt for _x in obs_storage])
                _baseline.px_range = np.mean([_x[i].px_range for _x in obs_storage])
                _baseline.px_diff = np.mean([_x[i].px_diff for _x in obs_storage])
                _baseline.is_init = True

            baseline.append(baseline_buffer)
        return baseline

    @classmethod
    def ema_baseline(cls, window: int, x: list[ObsArray]) -> list[ObsArray]:
        from .utils import EMA
        alpha = 2 / (window + 1)
        last_baseline = None
        baseline = []

        for obs_array in x:
            if not last_baseline:
                last_baseline = obs_array.__copy__()
                baseline.append(last_baseline)
                continue

            baseline_buffer = cls(n=obs_array.n, interval=obs_array.obs_interval)
            for i, _baseline in enumerate(baseline_buffer):
                _baseline: SingleObs
                _baseline.acc_vol = EMA.calculate_ema(value=obs_array[i].acc_vol, memory=last_baseline[i].acc_vol, alpha=alpha)
                _baseline.acc_amt = EMA.calculate_ema(value=obs_array[i].acc_amt, memory=last_baseline[i].acc_amt, alpha=alpha)
                _baseline.px_range = EMA.calculate_ema(value=obs_array[i].px_range, memory=last_baseline[i].px_range, alpha=alpha)
                _baseline.px_diff = EMA.calculate_ema(value=obs_array[i].px_diff, memory=last_baseline[i].px_diff, alpha=alpha)
                _baseline.is_init = True

            last_baseline = baseline_buffer
            baseline.append(last_baseline)

        return baseline

    def interpolate(self, baseline: ObsArray, threshold: float = 0.25) -> ObsArray:
        x_interpolated = self.__copy__()
        idx, res = divmod(self.ts, self.obs_interval)
        idx = int(idx)

        for i, (obs, baseline) in enumerate(zip(x_interpolated, baseline)):
            if i < idx:
                continue

            if i == idx and res > self.obs_interval * threshold:
                obs_weight, baseline_weight = res, self.obs_interval - res
                obs.acc_vol += (baseline_weight / self.obs_interval) * baseline.acc_vol
                obs.acc_amt += (baseline_weight / self.obs_interval) * baseline.acc_amt
                obs.px_range = obs.px_range * np.sqrt(self.obs_interval / obs_weight) if obs_weight else 0
                obs.px_diff = obs.px_diff * np.sqrt(self.obs_interval / obs_weight) if obs_weight else 0
            else:
                obs.acc_vol = baseline.acc_vol
                obs.acc_amt = baseline.acc_amt
                obs.px_range = baseline.px_range
                obs.px_diff = baseline.px_diff

        return x_interpolated

    def baseline_adjusted(self, baseline: ObsArray) -> ObsArray:
        adjusted_x = self.__copy__()

        for obs, obs_baseline, obs_adjusted in zip(self, baseline, adjusted_x):
            obs_adjusted.acc_vol = obs.acc_vol - obs_baseline.acc_vol
            obs_adjusted.acc_amt = obs.acc_amt - obs_baseline.acc_amt
            obs_adjusted.px_range = obs.px_range - obs_baseline.px_range
            obs_adjusted.px_diff = obs.px_diff - obs_baseline.px_diff

        return adjusted_x

    def __copy__(self):
        return self.__class__(
            n=self.n,
            interval=self.obs_interval,
            buffer=copy.copy(self.buffer)
        )

    @property
    def vol_ttl(self) -> float:
        return sum(_.acc_vol for _ in self)

    @property
    def amt_ttl(self) -> float:
        return sum(_.acc_amt for _ in self)

    @property
    def vol_arr(self) -> np.ndarray:
        return np.array([_.acc_vol for _ in self])

    @property
    def amt_arr(self) -> np.ndarray:
        return np.array([_.acc_amt for _ in self])


class VolumeProfile(object, metaclass=abc.ABCMeta):
    def __init__(self, ticker: str, interval: float = 5 * 60, profile: Profile = None, use_notional: bool = True):
        self.ticker = ticker
        self.obs_interval = interval
        self.profile = profile if profile is not None else GlobalStatics.PROFILE
        self.use_notional = use_notional

        self.x: dict[datetime.date, ObsArray] = {}
        self.baseline: dict[datetime.date, ObsArray] = {}
        self.current_x: ObsArray | None = None
        self.current_baseline: ObsArray | None = None

    @classmethod
    def _time_to_seconds(cls, t: datetime.time):
        return (t.hour * 60 + t.minute) * 60 + t.second + t.microsecond / 1000000

    def session_ts(self, t: datetime.time):
        """
        note that this assumes the t is not in any session break.
        """
        ts = self._time_to_seconds(t)
        session_ts = ts - self.session_start - sum(break_length for break_start, break_length in self.session_break.items() if break_start < ts)
        return session_ts

    @abc.abstractmethod
    def fit(self, data: Iterable[MarketData]):
        ...

    @abc.abstractmethod
    def predict(self, x: ObsArray = None):
        ...

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            profile=self.__class__.__name__,
            ticker=self.ticker,
            interval=self.obs_interval,
            use_notional=self.use_notional
        )

        if self.current_x is not None:
            data_dict['current_x'] = [(obs.to_list()) for obs in self.current_x]

        if self.current_baseline is not None:
            data_dict['current_baseline'] = [(obs.to_list()) for obs in self.current_baseline]

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        profile = globals()[json_message['profile']]
        self = profile.from_json(json_message)
        return self

    def clear(self):
        self.x.clear()
        self.baseline.clear()
        self.current_x = None
        self.current_baseline = None

    def on_data(self, market_data: MarketData):
        market_date, ts, idx = self.get_idx(market_data=market_data)

        if market_date in self.x:
            self.current_x = self.x[market_date]
        else:
            self.current_x = self.x[market_date] = ObsArray(n=self.n_interval, interval=self.obs_interval)

        self.current_x.update(market_data=market_data, ts=ts, idx=idx)

    def get_idx(self, market_data: MarketData) -> tuple[datetime.date, float, int]:
        market_time = market_data.market_time
        market_date = market_time.date()
        ts = self.session_ts(market_time.time())

        idx, res = divmod(ts, self.obs_interval)

        if idx and not res and isinstance(market_data, (TickData, BarData)):
            idx -= 1

        return market_date, ts, int(idx)

    @overload
    def _baseline_adjusted(self, x: dict[datetime.date, ObsArray] = None, baseline: dict[datetime.date, ObsArray] = None, **kwargs) -> dict[datetime.date, ObsArray]:
        ...

    @overload
    def _baseline_adjusted(self, x: ObsArray, baseline: ObsArray, **kwargs) -> ObsArray:
        ...

    def _baseline_adjusted(self, x=None, baseline=None, **kwargs):
        if x is None:
            x = self.x

        if self.baseline is None and not self.baseline:
            baseline_config = {name: value for name, value in kwargs.items() if (name in inspect.signature(self.generate_baseline).parameters and name != 'self')}
            baseline = self.generate_baseline(**baseline_config)

        match x:
            case ObsArray():
                if baseline is None:
                    baseline = self.current_baseline
                adjusted_x = x.baseline_adjusted(baseline=baseline)
            case dict():
                if baseline is None:
                    baseline = self.baseline
                adjusted_x = {market_date: self._baseline_adjusted(x=x[market_date], baseline=_baseline, **kwargs) for market_date, _baseline in baseline.items()}
            case _:
                raise TypeError('x and baseline must be a ObsArray or dict[date, Array[SingleObs]]!')

        return adjusted_x

    def generate_baseline(self, mode: Literal['sma', 'ema'] = 'ema', window: int = 5):
        match mode:
            case 'sma':
                baseline = ObsArray.sma_baseline(window=window, x=[self.x[_market_date] for _market_date in sorted(self.x)])
            case 'ema':
                baseline = ObsArray.ema_baseline(window=window, x=[self.x[_market_date] for _market_date in sorted(self.x)])
            case _:
                raise NotImplementedError(f'Invalid baseline mode {mode}!')

        # shift the baseline by one day
        baseline_out_sample = {}
        last_baseline = None
        for i, market_date in enumerate(sorted(self.x)):
            if last_baseline is None:
                last_baseline = baseline[i]
                continue

            baseline_out_sample[market_date] = last_baseline
            last_baseline = baseline[i]

        # update the attributes
        self.baseline.clear()
        self.baseline.update(baseline_out_sample)
        self.current_baseline = last_baseline

    @cached_property
    def session_start(self) -> float:
        if self.profile.session_start is None:
            return 0.

        return self._time_to_seconds(self.profile.session_start)

    @cached_property
    def session_end(self) -> float:
        if self.profile.session_end is None:
            return 24 * 60 * 60.

        return self._time_to_seconds(self.profile.session_end)

    @cached_property
    def session_break(self) -> dict[float, float]:
        if not self.profile.session_break:
            return {}

        breaks = {}
        for break_start_time, break_end_time in self.profile.session_break:
            start_ts = self._time_to_seconds(break_start_time)
            end_ts = self._time_to_seconds(break_end_time)
            breaks[start_ts] = end_ts - start_ts

        return breaks

    @cached_property
    def session_length(self) -> float:
        session_length = self.session_end - self.session_start - sum(break_ts for break_ts in self.session_break.values())
        return session_length

    @cached_property
    def n_interval(self) -> int:
        n, res = divmod(self.session_length, self.obs_interval)

        if res:
            LOGGER.error(f'{self.__class__.__name__} Can fully divides the session {self.session_length} into given interval {self.obs_interval}.')

        return int(n)

    @property
    def is_ready(self) -> bool:
        return False


class SimpleOnlineProfile(VolumeProfile):
    def __init__(self, ticker: str, n_window: int, interval: float = 60, profile: Profile = None, use_notional: bool = True):
        super().__init__(ticker=ticker, interval=interval, profile=profile, use_notional=use_notional)

        self._n_window = n_window
        self._ts = 0.
        self._y_pred = np.nan
        self._is_ready = False

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)

        data_dict.update(
            n_window=self._n_window,
            ts=self._ts,
            y_pred=self._y_pred,
            is_ready=self._is_ready
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        if isinstance(json_message, (str, bytes)):
            json_dict = json.loads(json_message)
        elif isinstance(json_message, dict):
            json_dict = json_message
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_message}')

        self = cls(
            ticker=json_dict['ticker'],
            interval=json_dict['interval'],
            n_window=json_dict['n_window']
        )

        self._ts = json_dict['ts']
        self._y_pred = json_dict['y_pred']
        self._is_ready = json_dict['is_ready']

        return self

    def fit(self, data: Iterable[MarketData]):
        raise NotImplementedError(f'{self.__class__.__name__} not needed to be fit!')

    def predict(self, x: ObsArray = None, min_obs: int = None) -> float:
        if self.is_ready:
            return self._y_pred

        min_obs = self._n_window / 2 if min_obs is None else min_obs

        # step 0: validate observation array
        valid_obs: list[SingleObs] = []
        for i in range(self.n_interval):
            obs = self.current_x[i]

            if not obs.is_init:
                LOGGER.debug(f'Baseline {self.ticker} {i} not fully initialized!')
                break

            valid_obs.append(obs)

        if len(valid_obs) < min_obs:
            return np.nan

        if self.use_notional:
            y_pred = self._y_pred = sum(_.acc_amt for _ in valid_obs) / self._ts * self.session_length
        else:
            y_pred = self._y_pred = sum(_.acc_vol for _ in valid_obs) / self._ts * self.session_length

        return y_pred

    def on_data(self, market_data: MarketData):
        if self.is_ready:
            return

        market_date, ts, idx = self.get_idx(market_data=market_data)

        if market_date in self.x:
            self.current_x = self.x[market_date]
        else:
            self.current_x = self.x[market_date] = ObsArray(n=self.n_interval, interval=self.obs_interval)

        if idx < self.n_interval:
            self._ts = ts
            self.current_x.update(market_data=market_data, ts=ts, idx=idx)
        else:
            self._is_ready = True
            return

    @property
    def n_interval(self) -> int:
        return self._n_window

    @property
    def is_ready(self) -> bool:
        return self._is_ready


class AccumulatedVolumeProfile(VolumeProfile):
    def __init__(self, ticker: str, interval: float = 5 * 60, profile: Profile = None, use_notional: bool = True):
        super(AccumulatedVolumeProfile, self).__init__(ticker=ticker, interval=interval, profile=profile, use_notional=use_notional)

        self._ts = 0.
        self._y_pred = np.nan
        self.curve: np.ndarray | None = None

        self.interpolation_threshold = 0.25  # minimal percentage of the input for interpolation

    def fit(self, data: Iterable[MarketData], **kwargs):
        if data:
            self.clear()

            for market_data in data:
                self.on_data(market_data=market_data)

        self.generate_baseline()
        x_adjusted = self._baseline_adjusted(x=self.x, **kwargs)
        x_array = np.array([_x.flatten() for market_date, _x in x_adjusted.items()])
        x_array = np.hstack([x_array, np.ones((x_array.shape[0], 1))])

        if self.use_notional:
            y_array = np.array([x_adjusted[market_date].amt_ttl for market_date in x_adjusted]).reshape(-1, 1)
        else:
            y_array = np.array([x_adjusted[market_date].vol_ttl for market_date in x_adjusted]).reshape(-1, 1)

        beta, *_ = np.linalg.lstsq(a=x_array, b=y_array, rcond=-1)
        self.curve = beta

    def predict(self, x: ObsArray = None) -> float:
        if x is None:
            x_obs = self.current_x
            baseline = self.current_baseline
            x_interpolated = x_obs.interpolate(baseline=baseline, threshold=self.interpolation_threshold)
            x_adjusted = self._baseline_adjusted(x=x_interpolated, baseline=baseline)
            x = x_adjusted

        x_array = np.append(x.flatten(), 1.)
        y = x_array @ self.curve
        if self.use_notional:
            self._y_pred = float(y[0]) + self.current_baseline.amt_ttl
        else:
            self._y_pred = float(y[0]) + self.current_baseline.vol_ttl
        return self._y_pred

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = super().to_json(fmt='dict', **kwargs)

        if self.curve is not None:
            data_dict['curve'] = self.curve.tolist()

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        if isinstance(json_message, (str, bytes)):
            json_dict = json.loads(json_message)
        elif isinstance(json_message, dict):
            json_dict = json_message
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_message}')

        self = cls(
            ticker=json_dict['ticker'],
            interval=json_dict['interval'],
        )

        if 'current_x' in json_dict:
            current_obs_data = json_dict['current_x']
            n = len(current_obs_data)
            self.current_obs = ObsArray(n=n, interval=self.obs_interval, buffer=(n * SingleObs)(*[SingleObs(*_) for _ in current_obs_data]))

        if 'current_baseline' in json_dict:
            current_baseline_data = json_dict['current_baseline']
            n = len(current_baseline_data)
            self.current_baseline = ObsArray(n=n, interval=self.obs_interval, buffer=(n * SingleObs)(*[SingleObs(*_) for _ in current_baseline_data]))

        if 'curve' in json_dict:
            curve_data = json_dict['curve']
            self.curve = np.array(curve_data).reshape(-1, 1)

        return self

    def clear(self):
        super().clear()

        self._ts = 0.
        self._y_pred = None
        self.curve = None

    def is_ready(self):
        if self.curve is None:
            return False
        return True


class IntervalVolumeProfile(AccumulatedVolumeProfile):
    def __init__(self, ticker: str, interval: float = 5 * 60, profile: Profile = None, use_notional: bool = True):
        super().__init__(ticker=ticker, interval=interval, profile=profile, use_notional=use_notional)

        self.curve: list[np.ndarray] = []

    def fit(self, data: Iterable[MarketData], **kwargs):
        if data:
            self.clear()

            for market_data in data:
                self.on_data(market_data=market_data)

        self.generate_baseline()
        x_adjusted = self._baseline_adjusted(**kwargs)
        beta = []

        for idx in range(1, self.n_interval):
            _x_array = np.array([_x[:idx].flatten() for market_date, _x in x_adjusted.items()])
            _x_array = np.hstack([_x_array, np.ones((_x_array.shape[0], 1))])
            if self.use_notional:
                _y_array = np.array([x_adjusted[market_date].amt_arr[idx] for market_date in x_adjusted]).reshape(-1, 1)
            else:
                _y_array = np.array([x_adjusted[market_date].vol_arr[idx] for market_date in x_adjusted]).reshape(-1, 1)

            _beta, *_ = np.linalg.lstsq(a=_x_array, b=_y_array, rcond=-1)
            beta.append(_beta)

        self.curve.clear()
        self.curve.extend(beta)

    def predict(self, x: ObsArray = None) -> float:
        if x is None:
            x_obs = self.current_x
            baseline = self.current_baseline
            x_interpolated = x_obs.interpolate(baseline=baseline, threshold=self.interpolation_threshold)
            x_adjusted = self._baseline_adjusted(x=x_interpolated, baseline=baseline)
            x = x_adjusted

        y = []
        idx, _ = divmod(self.current_x.ts, self.obs_interval)
        idx = int(idx)
        for i in range(self.n_interval):
            if i < idx:
                if self.use_notional:
                    y.append(self.current_x[i].acc_amt)
                else:
                    y.append(self.current_x[i].acc_vol)

                continue

            if i == 0:
                if self.use_notional:
                    y.append(self.current_baseline.amt_arr[0])
                else:
                    y.append(self.current_baseline.vol_arr[0])

                continue

            _x_array = np.append(x[:i].flatten(), 1.)
            _beta = self.curve[i - 1]
            if self.use_notional:
                _y = _x_array @ _beta + self.current_baseline.amt_arr[i]
            else:
                _y = _x_array @ _beta + self.current_baseline.vol_arr[i]
            y.append(float(_y[0]))

        self._y_pred = y
        return sum(self._y_pred)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = VolumeProfile.to_json(self=self, fmt='dict', **kwargs)

        if self.curve is not None:
            data_dict['curve'] = [_.tolist() for _ in self.curve]

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict):
        if isinstance(json_message, (str, bytes)):
            json_dict = json.loads(json_message)
        elif isinstance(json_message, dict):
            json_dict = json_message
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_message}')

        self = cls(
            ticker=json_dict['ticker'],
            interval=json_dict['interval'],
        )

        if 'current_x' in json_dict:
            current_obs_data = json_dict['current_x']
            n = len(current_obs_data)
            self.current_obs = ObsArray(n=n, interval=self.obs_interval, buffer=(n * SingleObs)(*[SingleObs(*_) for _ in current_obs_data]))

        if 'current_baseline' in json_dict:
            current_baseline_data = json_dict['current_baseline']
            n = len(current_baseline_data)
            self.current_baseline = ObsArray(n=n, interval=self.obs_interval, buffer=(n * SingleObs)(*[SingleObs(*_) for _ in current_baseline_data]))

        if 'curve' in json_dict:
            curve_data = json_dict['curve']
            self.curve = [np.array(_curve_data).reshape(-1, 1) for _curve_data in curve_data]

        return self

    def clear(self):
        VolumeProfile.clear(self=self)

        self._ts = 0.
        self._y_pred = None
        self.curve.clear()

    def is_ready(self):
        if not self.curve:
            return False
        return True
