import abc
import datetime
import enum
import math
from collections import defaultdict

import numpy as np
from PyQuantKit import MarketData

from . import LOGGER
from ..Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE
LOGGER = LOGGER.getChild('Decoder')


class WaveletFlag(enum.Enum):
    unknown = 'unknown'
    up = 'up'
    up_overshoot = 'up_overshoot'
    down = 'down'
    down_overshoot = 'down_overshoot'
    flat = 'flat'

    @property
    def is_up(self) -> bool:
        if self is self.up or self is self.up_overshoot:
            return True
        else:
            return False

    @property
    def is_down(self) -> bool:
        if self is self.down or self is self.down_overshoot:
            return True
        else:
            return False

    @classmethod
    def markov_valid(cls, last_flag, current_flag) -> bool:

        # up, down anf flat state have no requirement of previous state
        if current_flag is cls.up or current_flag is cls.down or current_flag is cls.flat:
            return True

        # up overshoot requires that previous state is up
        if current_flag is cls.up_overshoot and last_flag.is_up:
            return True

        # down overshoot requires that previous state is down
        if current_flag is cls.down_overshoot and last_flag.is_down:
            return True

        # any other scenario is invalid
        return False


class Wavelet(object):

    def __init__(self, market_price: float, timestamp: float, **kwargs):
        self.start_ts = kwargs.get('start_ts', timestamp)
        self.end_ts = kwargs.get('end_ts', timestamp)
        self.start_price = kwargs.get('start_price', market_price)
        self.end_price = kwargs.get('end_price', market_price)
        self.local_high = kwargs.get('local_high', (market_price, timestamp))
        self.secondary_high = kwargs.get('secondary_high', (market_price, timestamp))
        self.local_low = kwargs.get('local_low', (market_price, timestamp))
        self.secondary_low = kwargs.get('secondary_low', (market_price, timestamp))
        self.flag = kwargs.get('flag', WaveletFlag.unknown)

    def update(self, market_data: MarketData = None, market_price: float = None, timestamp: float = None):
        if market_price is None:
            market_price = market_data.market_price

        if timestamp is None:
            timestamp = market_data.timestamp

        if market_price > self.local_high[0]:
            self.local_high = (market_price, timestamp)
            self.secondary_low = (market_price, timestamp)

        if market_price > self.secondary_high[0]:
            self.secondary_high = (market_price, timestamp)

        if market_price < self.local_low[0]:
            self.local_low = (market_price, timestamp)
            self.secondary_high = (market_price, timestamp)

        if market_price < self.secondary_low[0]:
            self.secondary_low = (market_price, timestamp)

        self.end_price = market_price
        self.end_ts = timestamp

    @property
    def pct_change(self):
        return self.end_price / self.start_price - 1 if self.start_price else (math.copysign(np.inf, self.end_price) if self.end_price else 0.)


class Decoder(object, metaclass=abc.ABCMeta):
    """
    mark the market movement into different trend:
    each trend is summarized as a wavelet
    upon a new wavelet is confirmed, a callback function will be called. Callback function need to be registered.
    """

    def __init__(self):
        self.state_history: dict[str, list[Wavelet]] = {}
        self.current_state: dict[str, Wavelet] = {}
        self.callback: dict[str, list[callable]] = {}

    def register_callback(self, callback: callable, ticker: str = 'global'):

        if ticker in self.callback:
            callbacks = self.callback[ticker]
        else:
            callbacks = self.callback[ticker] = []

        if callback in callbacks:
            LOGGER.warning(f'callback {callback} redundant, is this intentional?')

        callbacks.append(callback)
        return callbacks

    def confirm_wavelet(self, ticker: str, wavelet: Wavelet):
        callbacks = self.callback.get(ticker, []) + self.callback.get('global', [])

        for callback in callbacks:
            callback(wavelet)

        if ticker in self.state_history:
            state_history = self.state_history[ticker]
        else:
            state_history = self.state_history[ticker] = []

        state_history.append(wavelet)

    def __call__(self, market_data: MarketData, **kwargs):
        ticker = market_data.ticker
        market_price = market_data.market_price
        timestamp = market_data.timestamp

        self.update_decoder(ticker=ticker, market_price=market_price, timestamp=timestamp)

    @abc.abstractmethod
    def update_decoder(self, ticker: str, market_price: float, timestamp: float):
        ...

    def clear(self):
        self.state_history.clear()
        self.current_state.clear()
        self.callback.clear()


class OnlineDecoder(Decoder):
    """
    online decoder:
    - when price goes up to 1% (up_threshold)
    - when price goes down to 1% (down_threshold)
    - when price goes up / down, relative to the local minimum / maximum, more than 0.5%  (confirmation_level)
    - when the market trend goes on more than 15 * 60 seconds (timeout)

    In retrospective mode, some wavelet can not be confirmed in realtime, the callback function is triggered later.

    By default, retrospective mode is set off. This module is focused on online decoding.
    """

    def __init__(self, confirmation_level: float = 0.005, timeout: float = 15 * 60, up_threshold: float = 0.01, down_threshold: float = 0.01, retrospective: bool = False):
        super().__init__()

        self.confirmation_level = confirmation_level
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.timeout = timeout
        self.retrospective = retrospective

    def update_decoder(self, ticker: str, market_price: float, timestamp: float):
        state_history = self.state_history.get(ticker)
        current_state = self.current_state.get(ticker)
        last_state = state_history[-1].flag if state_history else WaveletFlag.unknown

        if current_state is None:
            current_state = self.current_state[ticker] = Wavelet(market_price=market_price, timestamp=timestamp)

        # up overshoot
        if market_price > current_state.start_price * (1 + self.up_threshold) and (last_state.is_up or last_state is WaveletFlag.unknown):
            is_new_wavelet = True
            flag = WaveletFlag.up_overshoot
        # down state
        elif market_price < current_state.start_price * (1 - self.down_threshold) and (last_state.is_down or last_state is WaveletFlag.unknown):
            is_new_wavelet = True
            flag = WaveletFlag.down_overshoot
        # timeout
        elif timestamp > current_state.start_ts + self.timeout:
            is_new_wavelet = True
            flag = WaveletFlag.flat
        # up
        elif market_price < current_state.local_high[0] * (1 - self.confirmation_level) and not last_state.is_up:
            is_new_wavelet = True
            flag = WaveletFlag.down
        # down
        elif market_price > current_state.local_low[0] * (1 + self.confirmation_level) and not last_state.is_down:
            is_new_wavelet = True
            flag = WaveletFlag.up
        else:
            is_new_wavelet = False
            flag = WaveletFlag.unknown
            current_state.update(market_price=market_price, timestamp=timestamp)

        if is_new_wavelet:
            current_state.flag = flag
            new_state_params = {}

            if self.retrospective:
                if flag is WaveletFlag.up:
                    market_price = current_state.end_price = current_state.local_high[0]
                    timestamp = current_state.end_ts = current_state.local_high[1]
                    new_state_params['local_low'] = current_state.secondary_low
                elif flag is WaveletFlag.down:
                    market_price = current_state.end_price = current_state.local_low[0]
                    timestamp = current_state.end_ts = current_state.local_low[1]
                    new_state_params['local_high'] = current_state.secondary_high

            self.confirm_wavelet(ticker=ticker, wavelet=current_state)
            self.current_state[ticker] = Wavelet(market_price=market_price, timestamp=timestamp, **new_state_params)


class RecursiveDecoder(Decoder):
    """
    a modified wave decoder based on https://en.wikipedia.org/wiki/Elliott_wave_principle
    a recursive decoder marks the local minimal / maximal, and connect them to form each wavelet
    recursive decoder can only be used offline, as it is not realtime decoder.

    recursive decoder should be used recursively, to decode multiple level.
    recursive decoder only output 2 state, up and down
    for level = 0, we mark the local maximum / minimum of the market_price
    for level >= 1, we mark the local maximum of the maximums from previous level and minimums of the minimums from previous level
    """

    def __init__(self, level: int = 4):
        super().__init__()

        self.level = level

        self._local_maximum: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))  # the tuple contains the info of market_price, timestamp
        self._local_minimum: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        self._last_marking: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        self._last_maximum: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        self._last_minimum: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
        self._last_extreme: dict[str, dict[int, tuple[float, float, int]]] = defaultdict(dict)
        self._recursive_wavelet: dict[str, dict[int, list[Wavelet]]] = defaultdict(lambda: defaultdict(list))

    def update_decoder(self, ticker: str, market_price: float, timestamp: float):
        self._decode(ticker=ticker, market_price=market_price, timestamp=timestamp)

    def _decode(self, ticker: str, market_price: float, timestamp: float, **kwargs):
        flag = kwargs.get('flag', 0)
        level = kwargs.get('level', 0)

        local_maximum = self._local_maximum[ticker][level]
        local_minimum = self._local_minimum[ticker][level]
        last_marking = self._last_marking[ticker].get(level)
        last_marking_max = self._last_maximum[ticker].get(level)
        last_marking_min = self._last_minimum[ticker].get(level)
        recursive_wavelet = self._recursive_wavelet[ticker][level]

        assert level <= self.level

        # start the marking
        if not last_marking:
            self._last_marking[ticker][level] = (market_price, timestamp)
            recursive_wavelet.append(Wavelet(market_price=market_price, timestamp=timestamp))

            if level < self.level:
                self._decode(ticker=ticker, timestamp=timestamp, market_price=market_price, level=level + 1)
            else:
                self.current_state[ticker] = recursive_wavelet[-1]

            return

        current_wavelet = recursive_wavelet[-1]
        new_flag = 0

        if flag == 1:
            last_extreme = self._last_extreme[ticker].get(level)
            last_price = last_marking_max[0] if last_marking_max else last_marking[0]
            last_ts = last_marking_max[1] if last_marking_max else last_marking[1]
            self._last_maximum[ticker][level] = (market_price, timestamp)

            if last_marking_max is None or last_extreme is None or last_extreme[0] < last_price > market_price:
                local_maximum.append((last_price, last_ts))
                new_flag = 1
        elif flag == -1:
            last_extreme = self._last_extreme[ticker].get(level)
            last_price = last_marking_min[0] if last_marking_min else last_marking[0]
            last_ts = last_marking_min[1] if last_marking_min else last_marking[1]
            self._last_minimum[ticker][level] = (market_price, timestamp)

            if last_marking_min is None or last_extreme is None or last_extreme[0] > last_price < market_price:
                local_minimum.append((last_price, last_ts))
                new_flag = -1
        else:
            last_extreme = self._last_extreme[ticker].get(level)
            last_price = last_marking[0]
            last_ts = last_marking[1]

            if (last_extreme is None or last_extreme[2] == -1) and last_price > market_price:
                local_maximum.append((last_price, last_ts))
                new_flag = 1
            elif (last_extreme is None or last_extreme[2] == 1) and last_price < market_price:
                local_minimum.append((last_price, last_ts))
                new_flag = -1

        self._last_marking[ticker][level] = (market_price, timestamp)
        current_wavelet.update(market_price=market_price, timestamp=timestamp)

        if new_flag:
            self._last_extreme[ticker][level] = (last_price, last_ts, new_flag)

            if level < self.level:
                self._decode(ticker=ticker, market_price=last_price, timestamp=last_ts, flag=new_flag, level=level + 1)

            current_wavelet.flag = WaveletFlag.up if new_flag == 1 else WaveletFlag.down

            if level == self.level:
                self.confirm_wavelet(ticker=ticker, wavelet=current_wavelet)
                self.current_state[ticker] = Wavelet(market_price=market_price, timestamp=timestamp)

            recursive_wavelet.append(Wavelet(market_price=market_price, timestamp=timestamp))

    def plot(self, ticker: str):
        import plotly.graph_objects as go
        data = []

        for level in range(self.level + 1):
            local_maximum = self._local_maximum[ticker][level]
            local_minimum = self._local_minimum[ticker][level]
            local_extreme = local_maximum + local_minimum

            if not local_extreme:
                break

            local_extreme.sort(key=lambda _: _[1])
            y, x = zip(*local_extreme)
            x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in x]

            trace = go.Scatter(x=x, y=y, mode='lines', name=f'decode level {level}')
            data.append(trace)

        layout = go.Layout(title=f'Recursive Decoder {ticker}', xaxis=dict(title='X-axis'), yaxis=dict(title='Y-axis'))
        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes(rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")])
        return fig

    def local_extremes(self, ticker: str, level: int) -> list[tuple[float, float, int]]:
        local_maximum = [(_[0], _[1], 1) for _ in self._local_maximum[ticker][level]]
        local_minimum = [(_[0], _[1], -1) for _ in self._local_minimum[ticker][level]]
        local_extreme = local_maximum + local_minimum
        local_extreme.sort(key=lambda _: _[1])

        return local_extreme

    def clear(self):
        super().clear()

        self._local_maximum.clear()
        self._local_minimum.clear()
        self._last_marking.clear()
        self._last_maximum.clear()
        self._last_minimum.clear()
        self._last_extreme.clear()
        self._recursive_wavelet.clear()


__all__ = ['WaveletFlag', 'Wavelet', 'Decoder', 'OnlineDecoder', 'RecursiveDecoder']
