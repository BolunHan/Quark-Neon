import abc
from collections import deque

import numpy as np
from PyQuantKit import MarketData, TradeData, TransactionData, TickData, BarData

from .. import LOGGER


class IndexWeight(dict):
    def __init__(self, index_name: str, *args, **kwargs):
        self.index_name = index_name

        super().__init__(*args, **kwargs)

    def normalize(self):
        total_weight = sum(list(self.values()))

        if not total_weight:
            return

        for _ in self:
            self[_] /= total_weight

    @property
    def components(self) -> list[str]:
        return list(self.keys())


class EMA(object, metaclass=abc.ABCMeta):
    def __init__(self, discount_interval: float, alpha: float = None, window: int = None):
        self.discount_interval = discount_interval
        self.alpha = alpha if alpha else 1 - 2 / (window + 1)
        self.window = window if window else round(2 / (1 - alpha) - 1)

        if not (0 < alpha < 1):
            LOGGER.warning(f'{self.__class__.__name__} should have an alpha from 0 to 1')

        if discount_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive discount_interval')

        self._last_discount_ts: dict[str, float] = {}
        self._history: dict[str, dict[str, float]] = {}
        self._current: dict[str, dict[str, float]] = {}
        self._window: dict[str, dict[str, deque[float]]] = {}
        self.ema: dict[str, dict[str, float]] = {}

    def _register_ema(self, name):
        self._history[name] = {}
        self._current[name] = {}
        self._window[name] = {}
        _ = self.ema[name] = {}
        return _

    def _update_ema(self, ticker: str, timestamp: float = None, replace_na: float = np.nan, **update_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {update_data}')
                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = (timestamp // self.discount_interval )* self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)

        # update to current
        for entry_name in update_data:
            if entry_name in self._current:
                if np.isfinite(_ := update_data[entry_name]):
                    current = self._current[entry_name][ticker] = _
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = replace_na * self.alpha + current * (1 - self.alpha)
                    else:
                        self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

    def _accumulate_ema(self, ticker: str, timestamp: float = None, replace_na: float = np.nan, **accumulative_data: float):
        if timestamp:
            last_discount = self._last_discount_ts.get(ticker, 0.)

            if last_discount > timestamp:
                LOGGER.warning(f'{self.__class__.__name__} received obsolete {ticker} data! ts {timestamp}, data {accumulative_data}')

                time_span = max(0., last_discount - timestamp)
                adjust_factor = time_span // self.discount_interval
                alpha = self.alpha ** adjust_factor

                for entry_name in accumulative_data:
                    if entry_name in self._history:
                        if np.isfinite(_ := accumulative_data[entry_name]):
                            current = self._current[entry_name].get(ticker, 0.)
                            memory = self._history[entry_name][ticker] = self._history[entry_name].get(ticker, 0.) + _ * alpha

                            if memory is None:
                                self.ema[entry_name][ticker] = replace_na * self.alpha + current * (1 - self.alpha)
                            else:
                                self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

                return

            # assign a ts on first update
            if ticker not in self._last_discount_ts:
                self._last_discount_ts[ticker] = (timestamp // self.discount_interval) * self.discount_interval

            # self._discount_ema(ticker=ticker, timestamp=timestamp)
        # add to current
        for entry_name in accumulative_data:
            if entry_name in self._current:
                if np.isfinite(_ := accumulative_data[entry_name]):
                    current = self._current[entry_name][ticker] = self._current[entry_name].get(ticker, 0.) + _
                    memory = self._history[entry_name].get(ticker)

                    if memory is None:
                        self.ema[entry_name][ticker] = replace_na * self.alpha + current * (1 - self.alpha)
                    else:
                        self.ema[entry_name][ticker] = memory * self.alpha + current * (1 - self.alpha)

    def _discount_ema(self, ticker: str, timestamp: float):
        last_update = self._last_discount_ts.get(ticker, timestamp)

        # a discount event is triggered
        if last_update + self.discount_interval <= timestamp:
            time_span = timestamp - last_update
            adjust_power = int(time_span // self.discount_interval)

            for entry_name in self._history:
                current = self._current[entry_name].get(ticker)
                memory = self._history[entry_name].get(ticker)
                window: deque = self._window[entry_name].get(ticker, deque(maxlen=self.window))

                # pre-check: drop None or nan
                if current is None or not np.isfinite(current):
                    return

                # step 1: update window
                for _ in range(adjust_power - 1):
                    if window:
                        window.append(window[-1])

                window.append(current)

                # step 2: re-calculate memory if window is not full
                if len(window) < window.maxlen or memory is None:
                    memory = None

                    for _ in window:
                        if memory is None:
                            memory = _

                        memory = memory * self.alpha + _ * (1 - self.alpha)

                # step 3: calculate ema value by memory and current value
                ema = memory * self.alpha + current * (1 - self.alpha)

                # step 4: update EMA state
                self._current[entry_name].pop(ticker)
                self._history[entry_name][ticker] = ema
                self._window[entry_name][ticker] = window
                self.ema[entry_name][ticker] = ema

        self._last_discount_ts[ticker] = (timestamp // self.discount_interval) * self.discount_interval

    def _check_discontinuity(self, timestamp: float, tolerance: int = 1):
        discontinued = []

        for ticker in self._last_discount_ts:
            last_update = self._last_discount_ts[ticker]

            if last_update + (tolerance + 1) * self.discount_interval < timestamp:
                discontinued.append(ticker)

        return discontinued

    def _discount_all(self, timestamp: float):
        for _ in self._check_discontinuity(timestamp=timestamp, tolerance=1):
            self._discount_ema(ticker=_, timestamp=timestamp)

    def clear(self):
        self._last_discount_ts.clear()
        self._history.clear()
        self._current.clear()
        self._window.clear()
        self.ema.clear()


class Synthetic(object, metaclass=abc.ABCMeta):
    def __init__(self, weights: dict[str, float]):
        self.weights: IndexWeight = weights if isinstance(weights, IndexWeight) else IndexWeight(index_name='synthetic', **weights)
        self.weights.normalize()

        self.base_price: dict[str, float] = {}
        self.last_price: dict[str, float] = {}
        self.synthetic_base_price = 1.

    def composite(self, values: dict[str, float], replace_na: float = 0.):
        weighted_sum = 0.

        for ticker, weight in self.weights.items():
            value = values.get(ticker, replace_na)

            if np.isnan(value):
                weighted_sum += replace_na * self.weights[ticker]
            else:
                weighted_sum += value * self.weights[ticker]

        return weighted_sum

    def _update_synthetic(self, ticker: str, market_price: float):
        if ticker not in self.weights:
            return

        if ticker not in self.base_price:
            self.base_price[ticker] = market_price

        self.last_price[ticker] = market_price

    def clear(self):
        self.base_price.clear()
        self.last_price.clear()

    @property
    def synthetic_index(self):
        price_list = []
        weight_list = []

        for ticker in self.weights:
            weight_list.append(self.weights[ticker])

            if ticker in self.last_price:
                price_list.append(self.last_price[ticker] / self.base_price[ticker])
            else:
                price_list.append(1.)

        synthetic_index = np.average(price_list, weights=weight_list) * self.synthetic_base_price
        return synthetic_index

    @property
    def composited_index(self) -> float:
        return self.composite(self.last_price)


class FixTemporalIntervalMonitor(object, metaclass=abc.ABCMeta):

    def __init__(self, update_interval: float = 60., sample_interval: float = 1.):

        self.update_interval = update_interval
        self.sample_interval = sample_interval

        # Warning for update_interval
        if update_interval <= 0:
            LOGGER.warning(f'{self.__class__.__name__} should have a positive update_interval')

        # Warning for sample_interval by Shannon's Theorem
        if update_interval / 2 < sample_interval:
            LOGGER.warning(f"{self.__class__.__name__} should have a smaller sample_interval by Shannon's Theorem, max value {update_interval / 2}")

        # Error if sample_interval is not a fraction of update_interval
        if not (update_interval / sample_interval).is_integer():
            LOGGER.error(f"{self.__class__.__name__} should have a smaller sample_interval that is a fraction of the update_interval")

    def log_obs(self, ticker: str, value: float, timestamp: float, storage: dict[str, dict[float, float]]):
        if ticker in storage:
            sampled_obs = storage[ticker]
        else:
            sampled_obs = storage[ticker] = {}

        idx = timestamp // self.sample_interval
        idx_min = idx - self.update_interval // self.sample_interval

        if idx not in sampled_obs:
            sampled_obs[idx] = value
            self.on_entry_add(key=idx, value=value)
        else:
            sampled_obs[idx] = value
            self.on_entry_update(key=idx, value=value)

        for idx in list(sampled_obs):
            if idx < idx_min:
                sampled_obs.pop(idx)
                self.on_entry_pop(key=idx, value=value)
            else:
                break

    def on_entry_add(self, key, value):
        pass

    def on_entry_update(self, key, value):
        pass

    def on_entry_pop(self, key, value):
        pass

    def clear(self):
        pass


class FixVolumeIntervalMonitor(FixTemporalIntervalMonitor, metaclass=abc.ABCMeta):

    def __init__(self, update_interval: float = 100., sample_interval: float = 10.):
        super().__init__(update_interval=update_interval, sample_interval=sample_interval)
        self._accumulated_volume: dict[str, float] = {}

    def accumulate_volume(self, ticker: str = None, volume: float = 0., market_data: MarketData = None, use_notional: bool = False):

        if market_data is not None and isinstance(market_data, (TradeData, TransactionData)):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
        elif isinstance(market_data, TickData):
            ticker = market_data.ticker
            acc_volume = market_data.total_traded_notional if use_notional else market_data.total_traded_volume

            if acc_volume:
                self._accumulated_volume[ticker] = acc_volume
        elif isinstance(market_data, BarData):
            ticker = market_data.ticker
            volume = market_data.notional if use_notional else market_data.volume

            self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
        elif market_data is not None:
            raise NotImplementedError(f'Can not handle market data type {market_data.__class__}, expect TickData, BarData, TradeData and TransactionData.')
        else:
            if ticker is not None:
                self._accumulated_volume[ticker] = self._accumulated_volume.get(ticker, 0.) + volume
            else:
                raise ValueError('Must assign market_data, or ticker and volume')

    def log_obs(self, ticker: str, value: float, storage: dict[str, dict[float, float]], volume_accumulated: float = None):
        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        super().log_obs(ticker=ticker, value=value, timestamp=volume_accumulated, storage=storage)

    def clear(self):
        self._accumulated_volume.clear()


class AdaptiveVolumeIntervalMonitor(FixVolumeIntervalMonitor, metaclass=abc.ABCMeta):

    def __init__(self, update_interval: float = 60., sample_rate: float = 20, baseline_window: int = 5):
        super().__init__(update_interval=update_interval, sample_interval=update_interval / sample_rate)

        self.update_interval = update_interval
        self.sample_rate = sample_rate
        self.baseline_window = baseline_window

        self._accumulated_volume: dict[str, float] = {}
        self._volume_baseline = {
            'baseline': {},  # type: dict[str, float]
            'obs_start_acc_vol': {},  # type: dict[str, float],
            'obs_acc_vol': {},  # type: dict[str, dict[float,float]],
        }

    def _update_volume_baseline(self, ticker: str, timestamp: float, volume_accumulated: float = None) -> float | None:
        volume_baseline = self._volume_baseline['baseline']

        if ticker in volume_baseline:
            return volume_baseline[ticker]

        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        if ticker in (_ := self._volume_baseline['obs_acc_vol']):
            obs_acc_vol = _[ticker]
        else:
            obs_acc_vol = _[ticker] = {}

        if not obs_acc_vol:
            obs_start_acc_vol = self._volume_baseline['obs_start_acc_vol'][ticker] = volume_accumulated  # in this case, one obs of the trade data will be missed
        else:
            obs_start_acc_vol = self._volume_baseline['obs_start_acc_vol'][ticker]

        obs_timestamp = timestamp // self.update_interval * self.update_interval
        obs_start_ts = list(obs_acc_vol)[0] if obs_acc_vol else obs_timestamp
        obs_end_ts = obs_start_ts + self.baseline_window * self.update_interval

        if timestamp < obs_end_ts:
            obs_acc_vol[obs_timestamp] = volume_accumulated - obs_start_acc_vol
            baseline_ready = False
        elif len(obs_acc_vol) == self.baseline_window:
            baseline_ready = True
        else:
            LOGGER.error(f'{self.__class__.__name__} baseline validity check failed! {ticker} data missed, expect {self.baseline_window} observation, got {len(obs_acc_vol)}.')
            baseline_ready = False

        obs_vol = {}
        last_vol_acc = 0.
        for ts, vol_acc in obs_acc_vol.items():
            obs_vol[ts] = vol_acc - last_vol_acc
            last_vol_acc = vol_acc

        if len(obs_vol) == 1:
            # scale the observation
            if timestamp - obs_timestamp > self.update_interval * 0.5:
                baseline_est = obs_vol[obs_timestamp] / (timestamp - obs_timestamp) * self.update_interval
            # can not estimate any baseline
            else:
                baseline_est = None
        else:
            if baseline_ready:
                baseline_est = np.mean([obs_vol[ts] for ts in obs_vol])
            elif timestamp - obs_timestamp > self.update_interval * 0.5:
                baseline_est = np.mean([obs_vol[ts] if ts != obs_timestamp else obs_vol[ts] / (timestamp - ts) * self.update_interval for ts in obs_vol])
            else:
                baseline_est = np.mean([obs_vol[ts] for ts in obs_vol if ts != obs_timestamp])

        if baseline_ready:
            volume_baseline[ticker] = baseline_est

        return baseline_est

    def log_obs(self, ticker: str, value: float, storage: dict[str, dict[tuple[float, float], float]], volume_accumulated: float = None, timestamp: float = None, allow_oversampling: bool = False):
        """
        you should not assume the storage[ticker] having a length of self.sample_rate
        """
        if volume_accumulated is None:
            volume_accumulated = self._accumulated_volume.get(ticker, 0.)

        if ticker in storage:
            sampled_obs = storage[ticker]
        else:
            sampled_obs = storage[ticker] = {}

        if timestamp is None:
            raise ValueError(f'{self.__class__.__name__} requires a timestamp input!')

        update_interval = self._update_volume_baseline(ticker=ticker, timestamp=timestamp, volume_accumulated=volume_accumulated)

        # baseline still in generating process, fallback to fixed temporal interval mode
        if update_interval is None:
            pass
        elif update_interval <= 0 or not np.isfinite(update_interval):
            LOGGER.error(f'Invalid volume update interval for {ticker}, expect positive float, got {update_interval}')
            return

        if update_interval is None:
            idx_vol = 0
            idx_ts = timestamp // self.sample_interval
        else:
            volume_sample_interval = update_interval / self.sample_rate
            idx_vol = volume_accumulated // volume_sample_interval
            idx_ts = timestamp // self.sample_interval if allow_oversampling else 0.

        idx = (idx_ts, idx_vol)
        idx_vol_min = idx_vol - self.sample_rate

        if idx not in sampled_obs:
            sampled_obs[idx] = value
            self.on_entry_add(key=idx, value=value)
        else:
            sampled_obs[idx] = value
            self.on_entry_update(key=idx, value=value)

        for idx in list(sampled_obs):
            idx_ts, idx_vol = idx

            if idx_vol == 0:
                allow_oversampling = False
                break
            elif idx_vol < idx_vol_min:
                sampled_obs.pop(idx)
                self.on_entry_pop(key=idx, value=value)
            else:
                break

        if not allow_oversampling and (to_pop := len(sampled_obs) - self.sample_rate) > 0:
            for idx in list(sampled_obs)[:to_pop]:
                sampled_obs.pop(idx)
                self.on_entry_pop(key=idx, value=value)

    def clear(self):
        super().clear()

        self._volume_baseline['baseline'].clear()
        self._volume_baseline['obs_start_acc_vol'].clear()
        self._volume_baseline['obs_acc_vol'].clear()
