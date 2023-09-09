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
from .. import DecisionCore
from ...Base import GlobalStatics
from ...Strategy import RecursiveDecoder, StrategyMetric

TIME_ZONE = GlobalStatics.TIME_ZONE


class LinearRegressionCore(DecisionCore):
    def __init__(self, ticker: str, **kwargs):
        super().__init__()

        self.ticker = ticker
        self.data_source = kwargs.get('data_source', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res'))

        self.decision_params = SimpleNamespace(
            gain_threshold=kwargs.get('gain_threshold', 0.005),
            risk_threshold=kwargs.get('gain_threshold', 0.002)
        )
        self.calibration_params = SimpleNamespace(
            pred_length=kwargs.get('pred_length', 15 * 60),
            trace_back=kwargs.get('calibration_days', 5),  # use previous caches to train the model
        )

        self.inputs_var = ['TradeFlow.EMA.Sum', 'Coherence.Price.Up', 'Coherence.Price.Down',
                           'Coherence.Price.Ratio.EMA', 'Coherence.Volume', 'TA.MACD.Index',
                           'Aggressiveness.EMA.Net', 'Entropy.Price.EMA',
                           'Dummies.IsOpening', 'Dummies.IsClosing']
        self.pred_var = ['pct_chg']
        self.coefficients: pd.DataFrame | None = None

    def __str__(self):
        return f'DecisionCore.Linear.{self.__class__.__name__}(ready={self.is_ready})'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            ticker=self.ticker,
            data_source=str(self.data_source),
            inputs_var=self.inputs_var,
            pred_var=self.pred_var,
            decision_params=dict(
                gain_threshold=self.decision_params.gain_threshold,
                risk_threshold=self.decision_params.risk_threshold,
            ),
            calibration_params=dict(
                trace_back=self.calibration_params.trace_back
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

        self = cls(ticker=json_dict['ticker'])

        self.inputs_var.clear()
        self.inputs_var.extend(json_dict['inputs_var'])

        self.pred_var.clear()
        self.pred_var.extend(json_dict['pred_var'])

        self.data_source = pathlib.Path(json_dict['data_source'])
        self.decision_params.gain_threshold = json_dict['decision_params']['gain_threshold']
        self.decision_params.risk_threshold = json_dict['decision_params']['risk_threshold']
        self.calibration_params.trace_back = json_dict['calibration_params']['trace_back']

        if json_dict['coefficients']:
            self.coefficients = pd.DataFrame(json_dict['coefficients'])

        return self

    def signal(self, position: PositionManagementService, factor: dict[str, float], timestamp: float) -> int:

        if not self.is_ready:
            return 0

        prediction = self.predict(factor=factor, timestamp=timestamp)
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
        report = {'start_ts': time.time()}
        metric_info = None
        x_list, y_list = [], []

        if metric is not None:
            metric_info = metric.info
            _x, _y = self._prepare(factors=metric_info)
            x_list.append(_x)
            y_list.append(_y)

        # step 0: collect data from metric
        if factor_cache is not None:
            if isinstance(factor_cache, pd.DataFrame):
                cache_list = [factor_cache]
            else:
                cache_list = factor_cache

            for _ in cache_list:
                _x, _y = self._prepare(factors=_)
                x_list.append(_x)
                y_list.append(_y)

        # Optional step 1.1: load data from previous sessions
        if (trace_back := trace_back if trace_back is not None else self.calibration_params.trace_back) > 0:
            from Quark.Backtest.factor_pool import FACTOR_POOL

            caches = FACTOR_POOL.locate_caches(market_date=kwargs.get('market_date'), size=int(trace_back), exclude_current=True)

            for file_path in caches:
                info = self.load_info_from_csv(file_path=file_path)
                _x, _y = self._prepare(factors=info)

                x_list.append(_x)
                y_list.append(_y)

        x, y = pd.concat(x_list), pd.concat(y_list)

        # step 2: fit the model
        coefficients, residuals = self.fit(x, y)
        report.update(data_entries=x.shape, residuals=residuals)

        # step 3: store the coefficient matrix
        report.update(coefficient='\n' + self.coefficients.to_string())

        # step 4: validate matrix
        if metric_info is not None:
            self._validate(info=metric_info)

        # step 5: dump fig
        if metric_info is not None:
            fig = self.plot(info=metric_info)
            fig.write_html(file := os.path.realpath(f'{self}.html'))
            report.update(fig_dump=file, end_ts=time.time(), time_cost=f"{time.time() - report['start_ts']:,.3f}s")

        return report

    def clear(self):
        self.coefficients = None

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        self.session_dummies(timestamp=timestamp, inplace=factor)
        x = {_: factor[_] for _ in self.inputs_var}  # to ensure the order of input data
        x = self._generate_x_features(x)
        prediction = self._pred(x=x)
        return prediction

    def predict_batch(self, factor: pd.DataFrame):
        x = self._generate_x_features(factor)

        # Perform the prediction using the regression coefficients
        prediction = self._pred(x=x)

        return pd.DataFrame(prediction, columns=self.pred_var, index=x.index)

    def _pred(self, x: np.ndarray | pd.DataFrame | dict[str, float]) -> np.ndarray | dict[str, float]:
        if isinstance(x, np.ndarray):
            prediction = np.dot(x, self.coefficients)
        elif isinstance(x, pd.DataFrame):
            prediction = pd.DataFrame(data=0., index=x.index, columns=self.pred_var)
            for pred_name in self.pred_var:
                for var_name in self.coefficients.index:
                    prediction[pred_name] += x[var_name] * self.coefficients.at[var_name, pred_name]
        elif isinstance(x, dict):
            prediction = {_: 0. for _ in self.pred_var}

            for pred_name in self.pred_var:
                coefficient = self.coefficients[pred_name]
                for var_name in x:

                    if var_name not in coefficient.index:
                        continue

                    prediction[pred_name] += coefficient[var_name] * x[var_name]
        else:
            raise TypeError(f'Invalid x type {type(x)}')

        return prediction

    def _prepare(self, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

        factors['pct_chg'] = None  # Create a new column to store the percentage changes
        for ts, row in factors.iterrows():  # type: float, dict
            t0 = ts
            t1 = ts + self.calibration_params.pred_length

            closest_index = None
            for index in factors.index:
                if index >= t1:
                    closest_index = index
                    break

            if closest_index is None:
                continue

            # Find the closest index greater or equal to ts + window
            closest_index = factors.index[factors.index >= t1].min()

            # Get the prices at ts and ts + window
            p0 = row['SyntheticIndex.Price']
            p1 = factors.at[closest_index, 'SyntheticIndex.Price']

            # Calculate the percentage change and assign it to the 'pct_chg' column
            factors.at[t0, 'pct_chg'] = (p1 / p0) - 1

        factors['pct_chg'] = factors['pct_chg'].astype(float)

        # step 4: assigning dummies
        self.session_dummies(factors.index, inplace=factors)

        # step 5: generate inputs for linear regression
        x, y = self.generate_inputs(factors)

        return x, y

    def _validate(self, info: pd.DataFrame):
        prediction = self.predict_batch(info[self.inputs_var])
        info['pred'] = prediction['pct_chg']

    def generate_inputs(self, factors: pd.DataFrame):
        filtered = factors.dropna()

        x = filtered[self.inputs_var]  # to ensure the order of input data
        x = self._generate_x_features(x=x)
        y = filtered[self.pred_var]
        return x, y

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, input_cols: list[str] = None):
        if input_cols is None:
            input_cols = x.columns

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()

        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()
        coefficients, residuals = self._fit(x=x, y=y)
        self.coefficients = pd.DataFrame(data=coefficients, columns=self.pred_var, index=input_cols)
        return self.coefficients, residuals

    def _fit(self, x: np.ndarray, y: np.ndarray):
        coefficients, residuals, *_ = np.linalg.lstsq(x, y, rcond=None)
        return coefficients, residuals

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

        # Convert the dictionary to a DataFrame if the input is a DataFrame
        if isinstance(x, pd.DataFrame):
            x_matrix = pd.DataFrame(x_columns)
            x_matrix = x_matrix.loc[:, x_matrix.nunique() > 1]
            x_matrix["Bias"] = 1.

        else:
            x_matrix = x_columns
            x_matrix["Bias"] = 1.

        return x_matrix

    @classmethod
    def mark_factors(cls, factors: pd.DataFrame, start_ts: float, end_ts: float = None):

        # Step 1: Reverse the selected DataFrame
        if start_ts is None:
            info_selected = factors.loc[:end_ts][::-1]
        elif end_ts is None:
            info_selected = factors.loc[start_ts:][::-1]
        else:
            info_selected = factors.loc[start_ts:end_ts][::-1]

        # Step 2: Calculate cumulative minimum and maximum of "index_price"
        info_selected['local_max'] = info_selected['SyntheticIndex.Price'].cummax()
        info_selected['local_min'] = info_selected['SyntheticIndex.Price'].cummin()

        # Step 3: Merge the result back into the original DataFrame
        factors['local_max'].update(info_selected['local_max'])
        factors['local_min'].update(info_selected['local_min'])

        return factors

    @classmethod
    def load_info_from_csv(cls, file_path: str | pathlib.Path):
        df = pd.read_csv(file_path, index_col=0)
        # df.index = [_.to_pydatetime().replace(tzinfo=TIME_ZONE).timestamp() for _ in pd.to_datetime(df.index)]
        return df

    @classmethod
    def plot(cls, info: pd.DataFrame):
        import plotly.graph_objects as go

        fig = go.Figure()
        x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in info.index]

        fig.add_trace(go.Scatter(x=x, y=info["SyntheticIndex.Price"], mode='lines', name='Index Value'))
        fig.add_trace(go.Scatter(x=x, y=info["pct_chg"], mode='lines', name='pct_chg', yaxis='y2'))
        fig.add_trace(go.Scatter(x=x, y=info["pred"], mode='lines', name='pct_pred', yaxis='y2'))

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


class LinearDecodingCore(LinearRegressionCore):
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)

        self.decode_level = kwargs.get('decode_level', 4)
        self.smooth_params = SimpleNamespace(
            alpha=kwargs.get('smooth_alpha', 0.008),
            look_back=kwargs.get('smooth_look_back', 5 * 60)
        )

        self.decoder = RecursiveDecoder(level=self.decode_level)
        self.pred_var = ['up_smoothed', 'down_smoothed']

    def to_json(self, fmt='dict') -> dict | str:

        json_dict = super().to_json(fmt='dict')

        json_dict.update(
            decode_level=self.decode_level,
            smooth_params=dict(
                alpha=self.smooth_params.alpha,
                look_back=self.smooth_params.look_back
            )
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

        # noinspection PyTypeChecker
        self = super().from_json(json_dict)  # type: LinearDecodingCore

        if 'decode_level' in json_dict:
            self.decode_level = json_dict['decode_level']

        if 'smooth_params' in json_dict:
            self.smooth_params.alpha = json_dict['smooth_params']['alpha']
            self.smooth_params.look_back = json_dict['smooth_params']['look_back']

        return self

    def signal(self, position: PositionManagementService, factor: dict[str, float], timestamp: float) -> int:

        if not self.is_ready:
            return 0

        prediction = self.predict(factor=factor, timestamp=timestamp)
        pred_up = prediction['up_smoothed']
        pred_down = prediction['down_smoothed']

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
        elif exposure > 0 and (pred_up + pred_down < 0 or pred_down < -self.decision_params.risk_threshold):
            action = -1
        # logic 1.3: unwind short position when overall prediction is up, or risk is too high
        elif exposure < 0 and (pred_up + pred_down > 0 or pred_up > self.decision_params.risk_threshold):
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
        elif pred_up > self.decision_params.gain_threshold and pred_down > -self.decision_params.risk_threshold:
            action = 1
        # logic 2.3: open short position when gain is high and risk is low
        # logic 2.4: disable short opening for now, the short pred is not stable
        # elif pred_up < self.decision_params.risk_threshold and pred_down < -self.decision_params.gain_threshold:
        #     action = -1
        # logic 3.1: hold still if unwind condition is not triggered
        # logic 3.2: no action when open condition is not triggered
        # logic 3.3: no action if prediction is not valid (containing nan)
        # logic 3.4: no action if not in valid trading hours (in this scenario, every hour is valid trading hour), this logic can be overridden by strategy's closing / eod behaviors.
        else:
            action = 0

        return action

    def trade_volume(self, position: PositionManagementService, cash: float, margin: float, timestamp: float, signal: int) -> float:
        return 1.

    def clear(self):
        super().clear()
        self.decoder.clear()

    def _prepare(self, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # step 1: decode metric based on the
        self.decoder.clear()
        self.decode_index_price(factors=factors)

        # step 2: mark the ups and down for each data point
        local_extreme = self.decoder.local_extremes(ticker='Synthetic', level=self.decode_level)
        factors['local_max'] = np.nan
        factors['local_min'] = np.nan

        for i in range(len(local_extreme)):
            start_ts = local_extreme[i][1]
            end_ts = local_extreme[i + 1][1] if i + 1 < len(local_extreme) else None
            self.mark_factors(factors=factors, start_ts=start_ts, end_ts=end_ts)

        factors['up_actual'] = factors['local_max'] / factors['SyntheticIndex.Price'] - 1
        factors['down_actual'] = factors['local_min'] / factors['SyntheticIndex.Price'] - 1

        # step 3: smooth out the breaking points
        factors['up_smoothed'] = factors['up_actual']
        factors['down_smoothed'] = factors['down_actual']

        for i in range(len(local_extreme) - 1):
            previous_extreme = local_extreme[i - 1] if i > 0 else None
            break_point = local_extreme[i]
            next_extreme = local_extreme[i + 1]
            self.smooth_out(factors=factors, previous_extreme=previous_extreme, break_point=break_point, next_extreme=next_extreme)

        # step 4: assigning dummies
        self.session_dummies(factors.index, inplace=factors)

        # step 5: generate inputs for linear regression
        x, y = self.generate_inputs(factors)

        return x, y

    def _validate(self, info: pd.DataFrame):
        prediction = self.predict_batch(info[self.inputs_var])
        info['up_pred'] = prediction['up_smoothed']
        info['down_pred'] = prediction['down_smoothed']

    def decode_index_price(self, factors: pd.DataFrame):
        for _ in factors.iterrows():  # type: tuple[float, dict]
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

    def smooth_out(self, factors: pd.DataFrame, previous_extreme: tuple[float, float, int] | None, break_point: tuple[float, float, int], next_extreme: tuple[float, float, int]):
        # LOGGER.info(f'{self.__class__.__name__} smoothing actual up/down with {self.smooth_params}')
        look_back: float = self.smooth_params.look_back
        next_extreme_price = next_extreme[0]
        break_ts = break_point[1]
        break_type = break_point[2]
        start_ts = max(previous_extreme[1], break_ts - look_back) if previous_extreme else break_ts - look_back
        end_ts = break_ts

        smooth_range = factors.loc[start_ts:end_ts]

        if break_type == 1:  # approaching to local maximum, use downward profit is discontinuous, using "up_actual" to smooth out
            max_loss = (-smooth_range.up_actual[::-1]).cummin()[::-1]
            potential = (next_extreme_price / smooth_range['SyntheticIndex.Price'] - 1).clip(None, 0)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / self.smooth_params.alpha if _ < self.smooth_params.alpha else 0)
            smoothed = potential * hold_prob + smooth_range.down_actual * (1 - hold_prob)
            factors['down_smoothed'].update(smoothed)
        elif break_type == -1:
            max_loss = smooth_range.down_actual[::-1].cummin()[::-1]
            potential = (next_extreme_price / smooth_range['SyntheticIndex.Price'] - 1).clip(0, None)
            hold_prob = (-max_loss).apply(lambda _: 1 - _ / self.smooth_params.alpha if _ < self.smooth_params.alpha else 0)
            smoothed = potential * hold_prob + smooth_range.up_actual * (1 - hold_prob)
            factors['up_smoothed'].update(smoothed)
        else:
            return

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


class LogLinearCore(LinearDecodingCore):
    def __init__(self, ticker: str, **kwargs):
        super().__init__(ticker=ticker, **kwargs)
        self.pred_cutoff = 0.01

    @classmethod
    def drop_na(cls, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        x_valid_index = np.where(np.isfinite(x).all(axis=1))[0]
        y_valid_index = np.where(np.isfinite(y).all(axis=1))[0]

        # Merge the indices
        valid_index = np.intersect1d(x_valid_index, y_valid_index)
        x = x[valid_index]
        y = y[valid_index]

        return x, y

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, input_cols: list[str] = None):
        if input_cols is None:
            input_cols = x.columns

        y = np.log(np.abs(y))
        x, y = self.drop_na(x=x.astype(np.float64).to_numpy(), y=y.astype(np.float64).to_numpy())
        coefficients, residuals = self._fit(x=x, y=y)
        self.coefficients = pd.DataFrame(data=coefficients, columns=self.pred_var, index=input_cols)
        return self.coefficients, residuals

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        prediction = super().predict(factor, timestamp)

        # Apply exponential transformation to the prediction results
        prediction[self.pred_var[0]] = np.exp(prediction[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction[self.pred_var[1]])

        prediction = {key: np.nan if abs(value) > self.pred_cutoff else value for key, value in prediction.items()}

        return prediction

    def predict_batch(self, x: pd.DataFrame):
        prediction = super().predict_batch(x)
        prediction = prediction.astype(np.float64)
        # Apply exponential transformation to the prediction results
        prediction[self.pred_var[0]] = np.exp(prediction[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction[self.pred_var[1]])

        prediction = prediction.applymap(lambda value: np.nan if abs(value) > self.pred_cutoff else value)

        return prediction


__all__ = ['LinearRegressionCore', 'LinearDecodingCore', 'LogLinearCore']
