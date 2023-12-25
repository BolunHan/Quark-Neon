import datetime
import json
from types import SimpleNamespace

import pandas as pd
from AlgoEngine.Engine import PositionManagementService

from . import LOGGER
from .lore import LinearLore
from .. import DecisionCore
from ...Base import GlobalStatics
from ...Strategy import StrategyMetric

TIME_ZONE = GlobalStatics.TIME_ZONE


class LinearDecisionCore(DecisionCore):
    def __init__(self, ticker: str, **kwargs):
        super().__init__()
        self.ticker = ticker
        self.data_lore = LinearLore(ticker=ticker, **kwargs)

        self.decision_params = SimpleNamespace(
            gain_threshold=kwargs.get('gain_threshold', 0.005),
            risk_threshold=kwargs.get('gain_threshold', 0.002)
        )

    def __str__(self):
        return f'DecisionCore.Linear.{self.__class__.__name__}(ready={self.is_ready})'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            ticker=self.ticker,
            decision_params=dict(
                gain_threshold=self.decision_params.gain_threshold,
                risk_threshold=self.decision_params.risk_threshold,
            ),
            data_lore=self.data_lore.to_json(fmt='dict')
        )

        if fmt == 'dict':
            return json_dict
        else:
            return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_str}')

        self = cls(ticker=json_dict['ticker'])

        self.decision_params.gain_threshold = json_dict['decision_params']['gain_threshold']
        self.decision_params.risk_threshold = json_dict['decision_params']['risk_threshold']

        self.data_lore = LinearLore.from_json(json_dict['data_lore'])

        return self

    def signal(self, position: PositionManagementService, factor: dict[str, float], timestamp: float) -> int:

        if not self.is_ready:
            return 0

        prediction = self.data_lore.predict(factor=factor, timestamp=timestamp)
        pred = prediction['pct_chg']

        if position is None:
            LOGGER.warning('position not given, assuming no position. NOTE: Only gives empty position in BACKTEST mode!')
            exposure_volume = working_volume = 0
        else:
            exposure_volume = position.exposure_volume
            working_volume = position.working_volume

        exposure = exposure_volume.get(self.ticker, 0.)
        working_long = working_volume['Long'].get(self.ticker, 0.)
        working_short = working_volume['Short'].get(self.ticker, 0.)

        # condition 0: no more action when having working orders
        if working_long or working_short:
            return 0
        # logic 1.1: no winding, only unwind position
        # logic 1.2: unwind long position when overall prediction is down, or risk is too high
        elif exposure > 0 and (pred < self.decision_params.risk_threshold):
            action = -1
        # logic 1.3: unwind short position when overall prediction is up, or risk is too high
        elif exposure < 0 and (pred > -self.decision_params.risk_threshold):
            action = 1
        # logic 1.4: fully unwind if market is about to close
        elif exposure and datetime.datetime.fromtimestamp(timestamp).time() >= datetime.time(14, 55):
            action = -exposure
        # logic 2.1: only open position when no exposure
        elif exposure:
            action = 0
        # logic 2.1.1: only open position after 10:00
        elif datetime.datetime.fromtimestamp(timestamp).time() < datetime.time(10, 00):
            action = 0
        # logic 2.2: open long position when gain is high and risk is low
        elif pred > self.decision_params.gain_threshold:
            action = 1
        # logic 2.3: open short position when gain is high and risk is low
        # logic 2.4: disable short opening for now, the short pred is not stable
        elif pred < -self.decision_params.gain_threshold:
            action = -1
        # logic 3.1: hold still if unwind condition is not triggered
        # logic 3.2: no action when open condition is not triggered
        # logic 3.3: no action if prediction is not valid (containing nan)
        # logic 3.4: no action if not in valid trading hours (in this scenario, every hour is valid trading hour), this logic can be overridden by strategy's closing / eod behaviors.
        else:
            action = 0

        return action

    def trade_volume(self, position: PositionManagementService, cash: float, margin: float, timestamp: float, signal: int) -> float:
        return 1.

    def calibrate(self, metric: StrategyMetric = None, factor_cache: pd.DataFrame | list[pd.DataFrame] = None, trace_back: int = None, *args, **kwargs):
        report = self.data_lore.calibrate(metric=metric, factor_cache=factor_cache, trace_back=trace_back, *args, **kwargs)
        return report

    def clear(self):
        self.data_lore.clear()

    @property
    def is_ready(self):
        return self.data_lore.is_ready


__all__ = ['LinearDecisionCore']
