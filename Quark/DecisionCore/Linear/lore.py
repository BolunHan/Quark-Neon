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


def fix_prediction_target(factors: pd.DataFrame, pred_length: float) -> pd.Series:
    """
    Given a factor dataframe (StrategyMetric.info), return the prediction target, with fixed prediction length.

    This function does not take the market (session) breaks into account, as intended.
    And may return a series with Nan values.
    """
    target = dict()
    for ts, row in factors.iterrows():  # type: float, dict
        t0 = ts
        t1 = ts + pred_length

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
        target[t0] = (p1 / p0) - 1

    return pd.Series(target).astype(float)


class LinearLore(object):
    """
    Linear Data Lore

    This model is designed to handle the learning process of the linear decision core
    """

    def __init__(self, ticker: str, **kwargs):
        self.ticker = ticker

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
        return f'Lore.Linear.{self.ticker}'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            ticker=self.ticker,
            inputs_var=self.inputs_var,
            pred_var=self.pred_var,
            calibration_params=dict(
                trace_back=self.calibration_params.trace_back,
                pred_length=self.calibration_params.pred_length
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
        self.calibration_params.trace_back = json_dict['calibration_params']['trace_back']
        self.calibration_params.pred_length = json_dict['calibration_params']['pred_length']

        if json_dict['coefficients']:
            self.coefficients = pd.DataFrame(json_dict['coefficients'])

        return self

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

        # step 1: generating prediction target (y)
        factors['pct_chg'] = fix_prediction_target(
            factors=factors,
            pred_length=self.calibration_params.pred_length
        )

        # step 2: assigning session dummies
        self.session_dummies(factors.index, inplace=factors)

        # step 3: drop entries with nan
        filtered = factors[np.isfinite(factors).all(1)]
        # filtered = factors.dropna()
        if filtered.empty:
            raise ValueError('No valid training data found! All entries contain Nan')

        LOGGER.info(f'{1 - len(filtered) / len(factors):.2%} data is removed for nan or inf.')

        # step 3: generate inputs for linear regression
        x = filtered[self.inputs_var]  # to ensure the order of input data
        x = self._generate_x_features(x=x)
        y = filtered[self.pred_var]

        return x, y

    def _validate(self, info: pd.DataFrame):
        prediction = self.predict_batch(info[self.inputs_var])
        info['pred'] = prediction['pct_chg']

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
