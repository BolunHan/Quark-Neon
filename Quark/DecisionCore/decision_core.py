import datetime
import json

import numpy as np
import pandas as pd
from AlgoEngine.Engine import PositionManagementService

from . import LOGGER, DecisionCore
from ..Base import GlobalStatics
from ..DataLore import DataLore

TIME_ZONE = GlobalStatics.TIME_ZONE
__all__ = ['MajorityDecisionCore']


class MajorityDecisionCore(DecisionCore):
    def __init__(self, ticker: str, data_lore: DataLore, **kwargs):
        super().__init__()
        self.ticker = ticker
        self.data_lore = data_lore
        self.gain_threshold = kwargs.get('gain_threshold', 0.005)
        self.risk_threshold = kwargs.get('gain_threshold', 0.002)

    def __str__(self):
        return f'{self.__class__.__name__}.(ready={self.is_ready})'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            ticker=self.ticker,
            gain_threshold=self.gain_threshold,
            risk_threshold=self.risk_threshold,
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

        self = cls(
            ticker=json_dict['ticker'],
            data_lore=DataLore.from_json(json_dict['data_lore']),
            gain_threshold=json_dict['gain_threshold'],
            risk_threshold=json_dict['risk_threshold']
        )

        return self

    def predict(self, factor_value: dict[str, float], timestamp: float) -> dict[str, float]:
        if not self.is_ready:
            return {}

        prediction = self.data_lore.predict(factor_value=factor_value, timestamp=timestamp)
        return prediction

    def signal(self, position: PositionManagementService, prediction: dict[str, float], timestamp: float) -> int:
        if not self.is_ready:
            return 0

        signal_array = {}
        available_pred = set([_.split('.')[0] for _ in prediction.keys()])

        # handle single pred
        for pred_var in ['pct_change', 'target_actual', 'target_smoothed', 'state']:
            if pred_var not in available_pred:
                continue

            y_pred = prediction[pred_var]
            y_lower = prediction[f'{pred_var}.lower_bound']
            y_upper = prediction[f'{pred_var}.upper_bound']
            y_kelly = prediction.get(f'{pred_var}.kelly', 0.)

            if y_pred > 0:
                y_signal = self.signal_logic(
                    position=position,
                    expected_up=y_lower,
                    expected_down=y_lower,
                    kelly_proportion=y_kelly,
                    timestamp=timestamp
                )
            elif y_pred < 0:
                y_signal = self.signal_logic(
                    position=position,
                    expected_up=y_upper,
                    expected_down=y_upper,
                    kelly_proportion=y_kelly,
                    timestamp=timestamp
                )
            else:
                y_signal = 0

            signal_array[pred_var] = y_signal

        # handle paired pred
        for pred_var_up, pred_var_down in [('up_actual', 'down_actual'), ('up_smoothed', 'down_smoothed')]:
            if pred_var_up not in available_pred or pred_var_down not in available_pred:
                continue

            y_pred_up = prediction[pred_var_up]
            y_pred_up_lower = prediction[f'{pred_var_up}.lower_bound']
            y_pred_up_upper = prediction[f'{pred_var_up}.upper_bound']
            y_pred_down = prediction[pred_var_down]
            y_pred_down_lower = prediction[f'{pred_var_down}.lower_bound']
            y_pred_down_upper = prediction[f'{pred_var_down}.upper_bound']

            if y_pred_up + y_pred_down > 0:
                y_signal = self.signal_logic(
                    position=position,
                    expected_up=y_pred_up_lower,
                    expected_down=y_pred_down_lower,
                    kelly_proportion=0,
                    timestamp=timestamp
                )
            elif y_pred_up + y_pred_down < 0:
                y_signal = self.signal_logic(
                    position=position,
                    expected_up=y_pred_up_upper,
                    expected_down=y_pred_down_upper,
                    kelly_proportion=0,
                    timestamp=timestamp
                )
            else:
                y_signal = 0

            signal_array[pred_var] = y_signal

        # majority rules
        if np.sum(list(signal_array.values())) > 0.5:
            action = 1
        elif np.sum(list(signal_array.values())) < -0.5:
            action = -1
        else:
            action = 0.

        return action

    def signal_logic(self, position: PositionManagementService, expected_up: float, expected_down: float, kelly_proportion: float, timestamp: float) -> int:
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
        elif exposure > 0 and (expected_down < self.risk_threshold):
            action = -1
        # logic 1.3: unwind short position when overall prediction is up, or risk is too high
        elif exposure < 0 and (expected_up > -self.risk_threshold):
            action = 1
        # logic 1.4: fully unwind if market is about to close
        elif exposure and datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() >= datetime.time(14, 55):
            action = -exposure
        # logic 2.1: only open position when no exposure
        elif exposure:
            action = 0
        # logic 2.1.1: only open position after 10:00
        elif datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() < datetime.time(9, 35):
            action = 0
        # logic 2.2: open long position when gain is high and risk is low
        elif expected_up > self.gain_threshold or kelly_proportion > 0:
            action = 1
        # logic 2.3: open short position when gain is high and risk is low
        # logic 2.4: disable short opening for now, the short pred is not stable
        elif expected_down < -self.gain_threshold or kelly_proportion < 0:
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

    def calibrate(self, factor_value: pd.DataFrame, trace_back: int = None, market_date: datetime.date = None, **kwargs):
        factor_value_list = []

        if trace_back:
            from ..Factor.factor_pool import FACTOR_POOL
            caches = FACTOR_POOL.locate_caches(
                market_date=market_date,
                size=int(trace_back),
                exclude_current=True
            )

            for _ in caches:
                factor_value_list.append(pd.read_csv(_, index_col=0))

        factor_value_list.append(pd.DataFrame(factor_value).T)
        report = self.data_lore.calibrate(factor_value=factor_value_list, **kwargs)

        LOGGER.info(f'calibration report:\n' + '\n'.join([f"{_}: {report[_]}" for _ in report]))

        return report

    def clear(self):
        self.data_lore.clear()

    @property
    def is_ready(self):
        return self.data_lore.is_ready
