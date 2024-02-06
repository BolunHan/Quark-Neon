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
        self.prob_threshold = kwargs.get('prob_threshold', 0.8)

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

        if position is None:
            LOGGER.warning('position not given, assuming no position. NOTE: Only gives empty position in BACKTEST mode!')
            exposure_volume = working_volume = 0
        else:
            exposure_volume = position.exposure_volume
            working_volume = position.working_volume

        exposure = exposure_volume.get(self.ticker, 0.)
        working_long = working_volume['Long'].get(self.ticker, 0.)
        working_short = working_volume['Short'].get(self.ticker, 0.)

        signal_array = {}
        available_pred = set([_.split('.')[0] for _ in prediction.keys()])

        # handle single pred
        for pred_var in ['pct_change', 'target_actual', 'target_smoothed']:
            if pred_var not in available_pred:
                continue

            y_pred = prediction[pred_var]
            y_lower = prediction[f'{pred_var}.lower_bound']
            y_upper = prediction[f'{pred_var}.upper_bound']
            y_kelly = prediction.get(f'{pred_var}.kelly', 0.)

            if y_pred > 0:
                y_signal = self.signal_logic(
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=0.5,
                    prob_down=0.5,
                    expected_up=y_lower,
                    expected_down=y_lower,
                    kelly_proportion=y_kelly,
                    timestamp=timestamp
                )
            elif y_pred < 0:
                y_signal = self.signal_logic(
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=0.5,
                    prob_down=0.5,
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
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=0.5,
                    prob_down=0.5,
                    expected_up=y_pred_up_lower,
                    expected_down=y_pred_down_lower,
                    kelly_proportion=0,
                    timestamp=timestamp
                )
            elif y_pred_up + y_pred_down < 0:
                y_signal = self.signal_logic(
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=0.5,
                    prob_down=0.5,
                    expected_up=y_pred_up_upper,
                    expected_down=y_pred_down_upper,
                    kelly_proportion=0,
                    timestamp=timestamp
                )
            else:
                y_signal = 0

            signal_array[f'{pred_var_up}|{pred_var_down}'] = y_signal

        # handle single pred
        for pred_var in ['state']:
            if pred_var not in available_pred:
                continue

            y_pred = prediction[f'{pred_var}.lower_bound']
            y_lower = prediction[f'{pred_var}.lower_bound']
            y_upper = prediction[f'{pred_var}.upper_bound']

            prob_up = (y_pred + 1) / 2
            prob_up_lower = (y_lower + 1) / 2
            prob_up_upper = (y_upper + 1) / 2

            if prob_up > 0.5:
                y_signal = self.signal_logic(
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=prob_up_lower,
                    prob_down=1 - prob_up_lower,
                    expected_up=0,
                    expected_down=0,
                    kelly_proportion=0,
                    timestamp=timestamp
                )
            elif prob_up < 0.5:
                y_signal = self.signal_logic(
                    exposure=exposure,
                    working_long=working_long,
                    working_short=working_short,
                    prob_up=prob_up_upper,
                    prob_down=1 - prob_up_upper,
                    expected_up=0,
                    expected_down=0,
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

    def signal_logic(self, exposure: float, working_long: float, working_short: float, expected_up: float, expected_down: float, prob_up: float, prob_down: float, kelly_proportion: float, timestamp: float) -> int:
        # logic 0: single thread condition, no more action when having working orders
        if working_long or working_short:
            return 0

        # --- logic group 1: unwind rules ---

        # logic 1.1: no winding, only unwind position
        # logic 1.2: unwind long position when overall prediction is down, or risk is too high
        elif exposure > 0 and (expected_down < -self.risk_threshold) and kelly_proportion <= 0:
            action = -1
        # logic 1.3: unwind short position when overall prediction is up, or risk is too high
        elif exposure < 0 and (expected_up > self.risk_threshold) and kelly_proportion >= 0:
            action = 1
        # logic 1.4: unwind long position when up probability is small
        elif exposure > 0 and (prob_up < 0.5):
            action = -1
        # logic 1.5: unwind short position when down probability is small
        elif exposure < 0 and (prob_down < 0.5):
            action = 1
        # logic 1.6: fully unwind if market is about to close
        elif exposure and datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() >= datetime.time(14, 55):
            action = -exposure

        # --- logic group 2: maintain position ---

        # logic 2.1: only open position when no exposure
        elif exposure:
            action = 0
        # logic 2.2: only open position after 10:00
        elif datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() < datetime.time(9, 35):
            action = 0

        # --- logic group 3: open position ---

        # logic 3.1: open long position when gain is high and risk is low
        elif expected_up > self.gain_threshold or kelly_proportion > 0:
            action = 1
        # logic 3.2: open short position when gain is high and risk is low
        elif expected_down < -self.gain_threshold or kelly_proportion < 0:
            action = -1
        # logic 3.3: open long position when very likely to be in upward trend
        elif prob_up > self.prob_threshold:
            action = 1
        # logic 3.4: open long position when very likely to be in downward trend
        elif prob_down > self.prob_threshold:
            action = -1

        # --- logic group 4: misc ---

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

    def validation(self, factor_value: pd.DataFrame, **kwargs):
        return self.data_lore.validate(factor_value=pd.DataFrame(factor_value).T, **kwargs)

    def clear(self):
        self.data_lore.clear()

    @property
    def is_ready(self):
        return self.data_lore.is_ready
