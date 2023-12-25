import abc
from collections import deque
from functools import partial
from typing import Iterable

import numpy as np
from AlgoEngine.Engine import MarketDataMonitor, MDS

from .. import LOGGER
from ..Base import GlobalStatics

LOGGER = LOGGER.getChild('Factor')
TIME_ZONE = GlobalStatics.TIME_ZONE
DEBUG_MODE = GlobalStatics.DEBUG_MODE


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


class EMA(metaclass=abc.ABCMeta):
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
                self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

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
                self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

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

        self._last_discount_ts[ticker] = timestamp // self.discount_interval * self.discount_interval

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


class Synthetic(metaclass=abc.ABCMeta):
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


def add_monitor(monitor: MarketDataMonitor, **kwargs) -> dict[str, MarketDataMonitor]:
    monitors = kwargs.get('monitors', {})
    factors = kwargs.get('factors', None)
    register = kwargs.get('register', True)

    if factors is None:
        is_pass_check = True
    elif isinstance(factors, str) and factors == monitor.name:
        is_pass_check = True
    elif isinstance(factors, Iterable) and monitor.name in factors:
        is_pass_check = True
    else:
        return monitors

    if is_pass_check:
        monitors[monitor.name] = monitor

        if register:
            MDS.add_monitor(monitor)

    return monitors


ALPHA_05 = 0.9885  # alpha = 0.5 for each minute
ALPHA_02 = 0.9735  # alpha = 0.2 for each minute
ALPHA_01 = 0.9624  # alpha = 0.1 for each minute
ALPHA_001 = 0.9261  # alpha = 0.01 for each minute
ALPHA_0001 = 0.8913  # alpha = 0.001 for each minute
INDEX_WEIGHTS = IndexWeight(index_name='DummyIndex')

from .TradeFlow import *
from .Correlation import *
from .Misc import *
from .LowPass import *
from .Decoder import *


def register_monitor(**kwargs) -> dict[str, MarketDataMonitor]:
    monitors = kwargs.get('monitors', {})
    index_name = kwargs.get('index_name', 'SyntheticIndex')
    index_weights = IndexWeight(index_name=index_name, **kwargs.get('index_weights', INDEX_WEIGHTS))
    factors = kwargs.get('factors', None)

    index_weights.normalize()
    check_and_add = partial(add_monitor, factors=factors, monitors=monitors)
    LOGGER.info(f'Register monitors for index {index_name} and its {len(index_weights.components)} components!')

    # trade flow monitor
    check_and_add(TradeFlowMonitor())

    # trade flow ema monitor
    check_and_add(TradeFlowEMAMonitor(discount_interval=1, alpha=ALPHA_05, weights=index_weights))

    # price coherence monitor
    check_and_add(CoherenceMonitor(update_interval=60, sample_interval=1, weights=index_weights))

    # price coherence ema monitor
    check_and_add(CoherenceEMAMonitor(update_interval=60, sample_interval=1, weights=index_weights, discount_interval=1, alpha=ALPHA_0001))

    # trade coherence monitor
    check_and_add(TradeCoherenceMonitor(update_interval=60, sample_interval=1, weights=index_weights))

    # synthetic index monitor
    check_and_add(SyntheticIndexMonitor(index_name=index_name, weights=index_weights))

    # MACD monitor
    check_and_add(MACDMonitor(weights=index_weights, update_interval=60))

    # aggressiveness monitor
    check_and_add(AggressivenessMonitor())

    # aggressiveness ema monitor
    check_and_add(AggressivenessEMAMonitor(discount_interval=1, alpha=ALPHA_0001, weights=index_weights))

    # price coherence monitor
    check_and_add(EntropyMonitor(update_interval=60, sample_interval=1, weights=index_weights))

    # price coherence monitor
    check_and_add(EntropyEMAMonitor(update_interval=60, sample_interval=1, weights=index_weights, discount_interval=1, alpha=ALPHA_0001))

    # price coherence monitor
    check_and_add(VolatilityMonitor(weights=index_weights))

    # price movement online decoder
    check_and_add(DecoderMonitor(retrospective=False))

    # price movement online decoder
    check_and_add(IndexDecoderMonitor(up_threshold=0.005, down_threshold=0.005, confirmation_level=0.002, retrospective=True, weights=index_weights))

    return monitors


__all__ = [
    'LOGGER', 'TIME_ZONE', 'DEBUG_MODE', 'register_monitor', 'IndexWeight', 'Synthetic', 'EMA', 'register_monitor',
    # from .Correlation module
    'CoherenceMonitor', 'CoherenceEMAMonitor', 'TradeCoherenceMonitor', 'EntropyMonitor', 'EntropyEMAMonitor',
    # from Decoder module
    'DecoderMonitor', 'IndexDecoderMonitor', 'VolatilityMonitor',
    # from LowPass module
    'MACDMonitor', 'MACDTriggerMonitor', 'IndexMACDTriggerMonitor',
    # from Misc module
    'SyntheticIndexMonitor',
    # from TradeFlow module
    'AggressivenessMonitor', 'AggressivenessEMAMonitor', 'TradeFlowMonitor', 'TradeFlowEMAMonitor',
]
