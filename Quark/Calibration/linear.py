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
from ..Strategy import DecisionCore, RecursiveDecoder, StrategyMetric

TIME_ZONE = GlobalStatics.TIME_ZONE


class LinearCore(DecisionCore):
    def __init__(self, ticker: str, **kwargs):
        super().__init__()

        self.ticker = ticker

        self.decode_level = kwargs.get('decode_level', 4)
        self.data_source = kwargs.get('data_source', pathlib.Path(GlobalStatics.WORKING_DIRECTORY.value, 'Res'))
        self.smooth_params = SimpleNamespace(
            alpha=kwargs.get('smooth_alpha', 0.008),
            look_back=kwargs.get('smooth_look_back', 0.008)
        )
        self.decision_params = SimpleNamespace(
            gain_threshold=kwargs.get('gain_threshold', 0.008),
            risk_threshold=kwargs.get('gain_threshold', 0.004)
        )
        self.calibration_params = SimpleNamespace(
            trace_back=kwargs.get('calibration_days', 5),  # use previous caches to train the model
        )

        self.decoder = RecursiveDecoder(level=self.decode_level)
        self.inputs_var = ['TradeFlow.EMA.Sum', 'Coherence.Price.Up', 'Coherence.Price.Down',
                           'Coherence.Price.Ratio.EMA', 'Coherence.Volume', 'TA.MACD.Index',
                           'Aggressiveness.EMA.Net', 'Entropy.Price.EMA',
                           'Dummies.IsOpening', 'Dummies.IsClosing']
        self.pred_var = ['up_smoothed', 'down_smoothed']
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

        if position is None:
            LOGGER.warning('position not given, assuming no position. NOTE: Only gives empty position in BACKTEST mode!', norepeat=True)
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
            from ..Backtest.factor_pool import FACTOR_POOL

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
        # self.coefficients = pd.DataFrame(data=coefficients, columns=x.columns, index=self.pred_var).T
        report.update(coefficient='\n' + self.coefficients.to_string())

        # step 4: validate matrix
        if metric_info is not None:
            self._validate(info=metric_info)

        # step 5: dump fig
        if metric_info is not None:
            fig = self.plot(info=metric_info, decoder=self.decoder)
            fig.write_html(file := os.path.realpath(f'{self}.html'))
            report.update(fig_dump=file, end_ts=time.time(), time_cost=f"{time.time() - report['start_ts']:,.3f}s")

        return report

    def clear(self):
        self.decoder.clear()
        self.coefficients = None

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        self.session_dummies(timestamp=timestamp, inplace=factor)
        x = {_: factor[_] for _ in self.inputs_var}  # to ensure the order of input data
        x = self._generate_x_features(x)

        prediction = {_: 0. for _ in self.pred_var}

        for pred_name in self.pred_var:
            coefficient = self.coefficients[pred_name]
            for var_name in x:

                if var_name not in coefficient.index:
                    continue

                prediction[pred_name] += coefficient[var_name] * x[var_name]

        return prediction

    def predict_batch(self, x: pd.DataFrame):
        x = self._generate_x_features(x)

        # Perform the prediction using the regression coefficients
        prediction = np.dot(x, self.coefficients)

        return pd.DataFrame(prediction, columns=self.pred_var, index=x.index)

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

    def generate_inputs(self, factors: pd.DataFrame):
        filtered = factors.dropna()

        x = filtered[self.inputs_var]  # to ensure the order of input data
        x = self._generate_x_features(x=x)
        y = filtered[self.pred_var]
        return x, y

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        coefficients, residuals, *_ = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)
        self.coefficients = pd.DataFrame(data=coefficients.T, columns=x.columns, index=self.pred_var).T
        return self.coefficients, residuals

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

    @classmethod
    def drop_na(cls, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        x_valid_index = np.where(np.isfinite(x).all(axis=1))[0]
        y_valid_index = np.where(np.isfinite(y).all(axis=1))[0]

        # Merge the indices
        valid_index = np.intersect1d(x_valid_index, y_valid_index)
        x = x[valid_index]
        y = y[valid_index]

        return x, y

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        y[:, 0] = np.log(y[:, 0])
        y[:, 1] = np.log(-y[:, 1])

        x, y = self.drop_na(x=x, y=y)

        coefficients, residuals = super().fit(x, y)
        self.coefficients = pd.DataFrame(data=coefficients.T, columns=x.columns, index=self.pred_var).T
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


class RidgeCore(LogLinearCore):
    def __init__(self, ticker: str, **kwargs):
        self.alpha = kwargs.get('ridge_alpha', 1)

        super().__init__(ticker=ticker, **kwargs)

        self.scaler: pd.DataFrame | None = None
        self.pred_cutoff = 0.01

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = super().to_json(fmt='dict')

        if self.scaler is not None:
            json_dict['scaler'] = self.scaler.to_dict()

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

        self = super().from_json(json_dict)

        if 'scaler' in json_dict:
            self.scaler = pd.DataFrame(json_dict['scaler'])

        return self

    @classmethod
    def standardization_scaler(cls, x: pd.DataFrame):
        scaler = pd.DataFrame(index=['mean', 'std'], columns=x.columns)

        for col in x.columns:
            if col == 'Bias':
                scaler.loc['mean', col] = 0
                scaler.loc['std', col] = 1
            else:
                valid_values = x[col][np.isfinite(x[col])]
                scaler.loc['mean', col] = np.mean(valid_values)
                scaler.loc['std', col] = np.std(valid_values)

        return scaler

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        input_columns = x.columns
        scaler = self.standardization_scaler(x)
        x = (x - scaler.loc['mean']) / scaler.loc['std']
        y = np.log(np.abs(y))

        x, y = self.drop_na(x=x.astype(np.float64).to_numpy(), y=y.astype(np.float64).to_numpy())

        # Compute coefficients using ridge regression formula
        x_transpose = np.transpose(x)
        xtx = np.dot(x_transpose, x)
        xty = np.dot(x_transpose, y)
        lambda_identity = self.alpha * np.identity(len(xtx))
        coefficients = np.dot(np.linalg.inv(xtx + lambda_identity), xty)
        residuals = y - np.dot(x, coefficients)
        mse = np.mean(residuals ** 2, axis=0)

        self.scaler = scaler
        self.coefficients = pd.DataFrame(data=coefficients.T, columns=input_columns, index=self.pred_var).T
        return coefficients.T, mse

    def predict(self, factor: dict[str, float], timestamp: float) -> dict[str, float]:
        self.session_dummies(timestamp=timestamp, inplace=factor)
        x = {_: factor[_] for _ in self.inputs_var}  # to ensure the order of input data
        x = self._generate_x_features(x)

        prediction = {_: 0. for _ in self.pred_var}

        for pred_name in self.pred_var:
            coefficient = self.coefficients[pred_name]
            for var_name in x:

                if var_name not in coefficient.index:
                    continue

                prediction[pred_name] += coefficient[var_name] * (x[var_name] - self.scaler.at['mean', var_name]) / self.scaler.at['std', var_name]

        prediction[self.pred_var[0]] = np.exp(prediction[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction[self.pred_var[1]])

        for _ in prediction:
            if np.abs(prediction[_]) > self.pred_cutoff:
                prediction[_] = np.nan

        return prediction

    def predict_batch(self, x: pd.DataFrame):
        x = self._generate_x_features(x)

        # Perform the prediction using the regression coefficients
        prediction = np.dot((x - self.scaler.loc['mean']) / self.scaler.loc['std'], self.coefficients)
        prediction = pd.DataFrame(prediction, columns=self.pred_var, index=x.index)
        prediction[self.pred_var[0]] = np.exp(prediction.astype(np.float64)[self.pred_var[0]])
        prediction[self.pred_var[1]] = -np.exp(prediction.astype(np.float64)[self.pred_var[1]])

        prediction = prediction.applymap(lambda _: _ if -self.pred_cutoff < _ < self.pred_cutoff else np.nan)

        return prediction


__all__ = ['LinearCore', 'LogLinearCore', 'RidgeCore']
