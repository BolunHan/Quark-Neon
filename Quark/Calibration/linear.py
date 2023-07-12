__package__ = 'Quark.Calibration'

import datetime
import json
import os
import pathlib
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from AlgoEngine.Engine import PositionManagementService

from . import LOGGER
from ..Base import GlobalStatics
from ..Strategy.decision_core import DecisionCore
from ..Strategy.decoder import RecursiveDecoder
from ..Strategy.metric import StrategyMetric

TIME_ZONE = GlobalStatics.TIME_ZONE


class LinearCore(DecisionCore):
    def __init__(self, ticker: str, decode_level: int = 4, **kwargs):
        super().__init__()

        self.ticker = ticker
        self.decode_level = decode_level
        self.data_source = kwargs.get('data_source', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res'))

        self.decoder = RecursiveDecoder(level=self.decode_level)
        self.inputs_var = ['TradeFlow.EMA.Sum', 'Coherence.Price.Up', 'Coherence.Price.Down',
                           'Coherence.Price.Ratio.EMA', 'Coherence.Volume', 'TA.MACD.Index',
                           'Aggressiveness.EMA.Net', 'Entropy.Price.EMA',
                           'Dummies.IsOpening', 'Dummies.IsClosing']
        self.pred_var = ['up_smoothed', 'down_smoothed']
        self.smooth_params = SimpleNamespace(
            alpha=0.004,
            look_back=5 * 60
        )
        self.decision_params = SimpleNamespace(
            gain_threshold=0.005,
            risk_threshold=0.002
        )
        self.calibration_params = SimpleNamespace(
            look_back=5,
        )
        self.coefficients: pd.DataFrame | None = None

    def __str__(self):
        return f'DecisionCore.Linear.{id(self)}(ready={self.is_ready})'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            ticker=self.ticker,
            decode_level=self.decode_level,
            data_source=str(self.data_source),
            inputs_var=self.inputs_var,
            pred_var=self.pred_var,
            smooth_params=dict(
                alpha=self.smooth_params.alpha,
                look_back=self.smooth_params.look_back
            ),
            decision_params=dict(
                gain_threshold=self.decision_params.gain_threshold,
                risk_threshold=self.decision_params.risk_threshold,
            ),
            calibration_params=dict(
                look_back=self.calibration_params.look_back
            ),
            coefficients=self.coefficients.to_dict() if self.coefficients is not None else None
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
            decode_level=json_dict['decode_level'],
            data_source=pathlib.Path(json_dict['data_source'])
        )

        self.inputs_var.clear()
        self.inputs_var.extend(json_dict['inputs_var'])

        self.pred_var.clear()
        self.pred_var.extend(json_dict['pred_var'])

        self.smooth_params.alpha = json_dict['smooth_params']['alpha']
        self.smooth_params.look_back = json_dict['smooth_params']['look_back']
        self.decision_params.gain_threshold = json_dict['decision_params']['gain_threshold']
        self.decision_params.risk_threshold = json_dict['decision_params']['risk_threshold']
        self.calibration_params.look_back = json_dict['calibration_params']['look_back']

        if json_dict['coefficients']:
            self.coefficients = pd.DataFrame(json_dict['coefficients'])

        return self

    def signal(self, position: PositionManagementService, factor: dict[str, float], timestamp: float) -> int:

        if not self.is_ready:
            return 0

        prediction = self.predict(factor=factor, timestamp=timestamp)
        pred_up = prediction['up_smoothed']
        pred_down = prediction['down_smoothed']
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
        elif exposure > 0 and (pred_up + pred_down < 0 or pred_down < -self.decision_params.risk_threshold):
            action = -1
        # logic 1.3: unwind short position when overall prediction is up, or risk is too high
        elif exposure < 0 and (pred_up + pred_down > 0 or pred_up > self.decision_params.risk_threshold):
            action = 1
        # logic 2.1: only open position when no exposure
        # logic 2.2: open long position when gain is high and risk is low
        elif pred_up > self.decision_params.gain_threshold and pred_down > -self.decision_params.risk_threshold:
            action = 1
        # logic 2.3: open short position when gain is high and risk is low
        elif pred_up < self.decision_params.risk_threshold and pred_down < -self.decision_params.gain_threshold:
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

    def calibrate(self, metric: StrategyMetric = None, info: pd.DataFrame = None, *args, **kwargs):
        report = {'start_ts': time.time()}

        # step 0: collect data
        if info is None:
            info = metric.info

        LOGGER.info('Calibrating with factor data\n' + info.to_string())

        # step 1: prepare the data
        x, y = self._prepare(info=info)

        # Optional step 1.1: load data from previous sessions
        if (look_back := self.calibration_params.look_back) > 0:
            from ..Backtest.factor_pool import FACTOR_POOL

            x_list, y_list = [x], [y]
            caches = FACTOR_POOL.locate_caches(market_date=kwargs.get('market_date'), size=int(look_back), exclude_current=True)

            for file_path in caches:
                info = self.load_info_from_csv(file_path=file_path)
                _x, _y = self._prepare(info=info)

                x_list.append(_x)
                y_list.append(_y)
            x, y = pd.concat(x_list), pd.concat(y_list)

        # step 2: fit the model
        coefficients, residuals = self.fit(x.to_numpy(), y.to_numpy())
        report.update(data_entries=x.shape, residuals=residuals)

        # step 3: store the coefficient matrix
        self.coefficients = pd.DataFrame(data=coefficients, columns=x.columns, index=self.pred_var).T

        # step 4: validate matrix
        self._validate(info=info)
        report.update(coefficient='\n' + self.coefficients.to_string())

        # step 5: dump fig
        fig = self.plot(info=info, decoder=self.decoder)
        fig.write_html(file := os.path.realpath(f'{self}.html'))
        report.update(fig_dump=file, end_ts=time.time(), time_cost=f"{time.time() - report['start_ts']:,.3f}s")

        return report

    def clear(self):
        self.decoder.clear()
        self.coefficients = None

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        self.session_dummies(timestamp=timestamp, inplace=factor)
        x = {factor[_] for _ in self.inputs_var}  # to ensure the order of input data
        x = self._generate_x_features(x)

        prediction = {_: 0. for _ in self.pred_var}

        for pred_name in self.pred_var:
            coefficient = self.coefficients[pred_name]
            for var_name in x:
                prediction[pred_name] += coefficient[var_name] * x[var_name]

        return prediction

    def predict_batch(self, x: pd.DataFrame):
        x = self._generate_x_features(x)

        # Perform the prediction using the regression coefficients
        prediction = np.dot(x, self.coefficients)

        return pd.DataFrame(prediction, columns=self.pred_var, index=x.index)

    def _prepare(self, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # step 1: decode metric based on the
        self.decode_index_price(info=info)

        # step 2: mark the ups and down for each data point
        local_extreme = self.decoder.local_extremes(ticker='Synthetic', level=self.decode_level)
        info['local_max'] = np.nan
        info['local_min'] = np.nan

        for i in range(len(local_extreme)):
            start_ts = local_extreme[i][1]
            end_ts = local_extreme[i + 1][1] if i + 1 < len(local_extreme) else None
            self.mark_info(info=info, start_ts=start_ts, end_ts=end_ts)

        info['up_actual'] = info['local_max'] / info['SyntheticIndex.Price'] - 1
        info['down_actual'] = info['local_min'] / info['SyntheticIndex.Price'] - 1

        # step 3: smooth out the breaking points
        info['up_smoothed'] = info['up_actual']
        info['down_smoothed'] = info['down_actual']

        for i in range(len(local_extreme) - 1):
            previous_extreme = local_extreme[i - 1] if i > 0 else None
            break_point = local_extreme[i]
            next_extreme = local_extreme[i + 1]
            self.smooth_out(info=info, previous_extreme=previous_extreme, break_point=break_point, next_extreme=next_extreme)

        # step 4: assigning dummies
        self.session_dummies(info.index, inplace=info)

        # step 5: generate inputs for linear regression
        x, y = self.generate_inputs(info)

        return x, y

    def _validate(self, info: pd.DataFrame):
        prediction = self.predict_batch(info[self.inputs_var])
        info['up_pred'] = prediction['up_smoothed']
        info['down_pred'] = prediction['down_smoothed']

    def decode_index_price(self, info: pd.DataFrame):
        for _ in info.iterrows():  # type: tuple[float, dict]
            ts, row = _
            market_price = float(row.get('SyntheticIndex.Price', np.nan))
            market_time = datetime.datetime.fromtimestamp(ts, tz=TIME_ZONE)
            timestamp = market_time.timestamp()

            # filter nan values
            if not np.isfinite(market_price):
                continue

            # filter non-trading hours
            if market_time.time() < datetime.time(9, 30) \
                    or datetime.time(11, 30) < market_time.time() < datetime.time(13, 0) \
                    or datetime.time(15, 0) < market_time.time():
                continue

            self.decoder.update_decoder(ticker='Synthetic', market_price=market_price, timestamp=timestamp)

    def smooth_out(self, info: pd.DataFrame, previous_extreme: tuple[float, float, int] | None, break_point: tuple[float, float, int], next_extreme: tuple[float, float, int]):
        look_back: float = self.smooth_params.look_back
        next_extreme_price = next_extreme[0]
        break_ts = break_point[1]
        break_type = break_point[2]
        start_ts = max(previous_extreme[1], break_ts - look_back) if previous_extreme else break_ts - look_back
        end_ts = break_ts

        smooth_range = info.loc[start_ts:end_ts]

        if break_type == 1:  # approaching to local maximum, use downward profit is discontinuous, using "up_actual" to smooth out
            max_loss = (-smooth_range.up_actual[::-1]).cummin()[::-1]
            potential = (next_extreme_price / smooth_range['SyntheticIndex.Price'] - 1).clip(None, 0)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / self.smooth_params.alpha if _ < self.smooth_params.alpha else 0)
            smoothed = potential * hold_prob + smooth_range.down_actual * (1 - hold_prob)
            info['down_smoothed'].update(smoothed)
        elif break_type == -1:
            max_loss = smooth_range.down_actual[::-1].cummin()[::-1]
            potential = (next_extreme_price / smooth_range['SyntheticIndex.Price'] - 1).clip(0, None)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / self.smooth_params.alpha if _ < self.smooth_params.alpha else 0)
            smoothed = potential * hold_prob + smooth_range.up_actual * (1 - hold_prob)
            info['up_smoothed'].update(smoothed)
        else:
            return

    def generate_inputs(self, info: pd.DataFrame):
        filtered = info.dropna()

        x = filtered[self.inputs_var]  # to ensure the order of input data
        x = self._generate_x_features(x=x)
        y = filtered[self.pred_var]
        return x, y

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray):
        coefficients, residuals, *_ = np.linalg.lstsq(x, y, rcond=None)
        return coefficients.T, residuals

    @classmethod
    def session_dummies(cls, timestamp: float | list[float], inplace: dict[str, float | list[float]] | pd.DataFrame = None):
        d = {} if inplace is None else inplace

        if isinstance(timestamp, (float, int)):
            d['Dummies.IsOpening'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() < datetime.time(10, 30) else 0
            d['Dummies.IsClosing'] = 1 if datetime.datetime.fromtimestamp(timestamp, tz=TIME_ZONE).time() > datetime.time(14, 30) else 0
        else:
            d['Dummies.IsOpening'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() < datetime.time(10, 30) else 0 for _ in timestamp]
            d['Dummies.IsClosing'] = [1 if datetime.datetime.fromtimestamp(_, tz=TIME_ZONE).time() > datetime.time(14, 30) else 0 for _ in timestamp]

        return d

    @classmethod
    def _generate_x_features(cls, x: pd.DataFrame | dict) -> pd.DataFrame | dict:
        keys = list(x.keys())

        # Initialize an empty dictionary to store the X matrix columns
        x_columns = {}

        for key in keys:
            x_columns[key] = x[key]

        # Append squared sequences for all variables
        for key in keys:
            x_columns[key + "^2"] = np.power(x[key], 2)

        # Append interaction sequences for all variable combinations
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                x_columns[keys[i] + " * " + keys[j]] = x[keys[i]] * x[keys[j]]

        # Append bias sequence (filled with 1)
        x_columns["Bias"] = 1

        # Convert the dictionary to a DataFrame if the input is a DataFrame
        if isinstance(x, pd.DataFrame):
            x_matrix = pd.DataFrame(x_columns)
        else:
            x_matrix = x_columns

        return x_matrix

    @classmethod
    def mark_info(cls, info: pd.DataFrame, start_ts: float, end_ts: float = None):

        # Step 1: Reverse the selected DataFrame
        if start_ts is None:
            info_selected = info.loc[:end_ts][::-1]
        elif end_ts is None:
            info_selected = info.loc[start_ts:][::-1]
        else:
            info_selected = info.loc[start_ts:end_ts][::-1]

        # Step 2: Calculate cumulative minimum and maximum of "index_price"
        info_selected['local_max'] = info_selected['SyntheticIndex.Price'].cummax()
        info_selected['local_min'] = info_selected['SyntheticIndex.Price'].cummin()

        # Step 3: Merge the result back into the original DataFrame
        info['local_max'].update(info_selected['local_max'])
        info['local_min'].update(info_selected['local_min'])

        return info

    @classmethod
    def load_info_from_csv(cls, file_path: str | pathlib.Path):
        df = pd.read_csv(file_path, index_col=0)
        # df.index = [_.to_pydatetime().replace(tzinfo=TIME_ZONE).timestamp() for _ in pd.to_datetime(df.index)]
        return df

    @classmethod
    def plot(cls, info: pd.DataFrame, decoder: RecursiveDecoder = None):
        import plotly.graph_objects as go

        fig = go.Figure()
        x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in info.index]

        fig.add_trace(go.Scatter(x=x, y=info["SyntheticIndex.Price"], mode='lines', name='Index Value'))
        fig.add_trace(go.Scatter(x=x, y=info["local_max"], mode='lines', name='Up'))
        fig.add_trace(go.Scatter(x=x, y=info["local_min"], mode='lines', name='Down'))
        fig.add_trace(go.Scatter(x=x, y=info["up_actual"], mode='lines', name='up_actual', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["down_actual"], mode='lines', name='down_actual', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["up_smoothed"], mode='lines', name='up_smoothed', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["down_smoothed"], mode='lines', name='down_smoothed', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["up_pred"], mode='lines', name='up_pred', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["down_pred"], mode='lines', name='down_pred', yaxis='y2'))

        if decoder:
            for _ in decoder.plot('Synthetic').data:
                fig.add_trace(_)

        fig.update_layout(
            title=f'Linear action core',
            xaxis=dict(title='X-axis'),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Percentage", overlaying="y", side="right"),
        )

        fig.update_xaxes(rangebreaks=[dict(bounds=[11.5, 13], pattern="hour"), dict(bounds=[15, 9.5], pattern="hour")])
        return fig

    @property
    def is_ready(self):
        if self.coefficients is None:
            return False

        return True


class LogLinearCore(LinearCore):
    def fit(self, x: np.ndarray, y: np.ndarray):
        # Apply log transformation to the first column of y array
        y[:, 0] = np.log(y[:, 0])
        # Apply log transformation with negative sign to the second column of y array
        y[:, 1] = np.log(-y[:, 1])

        # Drop rows with NaN values in both x and y arrays
        valid_rows = np.logical_not(np.isnan(x).any(axis=1)) & np.logical_not(np.isnan(y).any(axis=1))
        x = x[valid_rows]
        y = y[valid_rows]

        # Perform the linear regression
        coefficients, residuals = super().fit(x, y)
        return coefficients, residuals

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        prediction = super().predict(factor, timestamp)

        # Apply exponential transformation to the prediction results
        prediction[self.pred_var[0]] = np.exp(prediction[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction[self.pred_var[1]])

        return prediction

    def predict_batch(self, x: pd.DataFrame):
        prediction = super().predict_batch(x)

        # Apply exponential transformation to the prediction results
        prediction[self.pred_var[0]] = np.exp(prediction[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction[self.pred_var[1]])

        return prediction
