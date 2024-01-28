import datetime
import json
import time

import numpy as np
import pandas as pd

from . import LOGGER, DataLore
from .utils import define_inputs, define_prediction
from ..Base import GlobalStatics
from ..Calibration import Regression
from ..Calibration.Kernel import SigmoidTransformer, BinaryTransformer
from ..Calibration.Linear.bootstrap import RidgeRegression, RidgeLogRegression, RidgeLogisticRegression
from ..Calibration.cross_validation import CrossValidation
from ..Calibration.kelly import kelly_bootstrap
from ..Factor.decoder import RecursiveDecoder

TIME_ZONE = GlobalStatics.TIME_ZONE
__all__ = ['LinearDataLore']


class LinearDataLore(DataLore):
    """
    Linear Data Lore

    This model is designed to handle the learning and predicting process of the linear data lore
    """

    def __init__(self, ticker: str, alpha: float, trade_cost: float, **kwargs):
        self.ticker = ticker
        self.alpha = alpha
        self.trade_cost = trade_cost
        self.decode_level = kwargs.get('decode_level', 3)
        self.poly_degree = kwargs.get('poly_degree', 2)
        self.pred_length = kwargs.get('pred_length', 15 * 60)
        self.bootstrap_samples = kwargs.get('bootstrap_samples', 100)
        self.bootstrap_block_size = kwargs.get('bootstrap_samples', 0.05)

        # this is an example of what input_var looks like
        self.inputs_var = ['Skewness.PricePct.Index.Adaptive.Index',
                           'Skewness.PricePct.Index.Adaptive.Slope',
                           'Gini.PricePct.Index.Adaptive',
                           'Coherence.Price.Adaptive.up',
                           'Coherence.Price.Adaptive.down',
                           'Coherence.Price.Adaptive.ratio',
                           'Coherence.Volume.up',
                           'Coherence.Volume.down',
                           'Entropy.Price.Adaptive',
                           'Entropy.Price',
                           'Entropy.PricePct.Adaptive',
                           'EMA.Divergence.Index.Adaptive.Index',
                           'EMA.Divergence.Index.Adaptive.Diff',
                           'EMA.Divergence.Index.Adaptive.Diff.EMA',
                           'TradeFlow.EMA.Index',
                           'Aggressiveness.EMA.Index']
        self.pred_var = ['pct_change',
                         'up_actual', 'down_actual', 'target_actual',
                         'up_smoothed', 'down_smoothed', 'target_smoothed',
                         'state']

        self.model: dict[str, Regression] = {}
        self.decoder = RecursiveDecoder(level=self.decode_level)
        self.cv = {}

    def _init_model(self, pred_var: str, y: np.ndarray = None):
        if pred_var in ['up_actual', 'up_smoothed']:
            if y is None:
                upper_bound = 0.02
            else:
                upper_bound = 2 * np.nanmean(y)

            model = RidgeLogRegression(
                alpha=0.2,
                transformer=SigmoidTransformer(lower_bound=0., upper_bound=upper_bound),
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_block_size=self.bootstrap_samples
            )
        elif pred_var in ['down_actual', 'down_smoothed']:
            if y is None:
                lower_bound = -0.02
            else:
                lower_bound = 2 * np.nanmean(y)

            model = RidgeLogRegression(
                alpha=0.2,
                transformer=SigmoidTransformer(lower_bound=lower_bound, upper_bound=0.),
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_block_size=self.bootstrap_samples
            )
        elif pred_var in ['state']:
            model = RidgeLogisticRegression(
                alpha=0.2,
                transformer=BinaryTransformer(center=0, scale=1.),
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_block_size=self.bootstrap_samples
            )
        else:
            model = RidgeRegression(
                alpha=0.2,
                bootstrap_samples=self.bootstrap_samples,
                bootstrap_block_size=self.bootstrap_samples
            )

        self.model[pred_var] = model
        return model

    def __str__(self):
        return f'Lore.Linear.{self.ticker}'

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            type=f'{self.__class__.__name__}',
            ticker=self.ticker,
            alpha=self.alpha,
            trade_cost=self.trade_cost,
            poly_degree=self.poly_degree,
            pred_length=self.pred_length,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_block_size=self.bootstrap_block_size,
            inputs_var=self.inputs_var,
            pred_var=self.pred_var,
            model={_: self.model[_].to_json(fmt='dict') for _ in self.model}
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
            alpha=json_dict['alpha'],
            trade_cost=json_dict['trade_cost'],
            poly_degree=json_dict['poly_degree'],
            pred_length=json_dict['pred_length'],
            bootstrap_samples=json_dict['bootstrap_samples'],
            bootstrap_block_size=json_dict['bootstrap_samples'],
        )

        self.inputs_var.clear()
        self.inputs_var.extend(json_dict['inputs_var'])

        self.pred_var.clear()
        self.pred_var.extend(json_dict['pred_var'])

        self.model.update({key: Regression.from_json(value) for key, value in json_dict['model'].items()})
        return self

    def calibrate(self, factor_value: pd.DataFrame | list[pd.DataFrame], *args, **kwargs):
        report = {'start_ts': time.time()}

        if isinstance(factor_value, pd.DataFrame):
            factor_value = [factor_value]

        x_train, y_train, x_val, y_val = [], [], None, None

        # define x, y inputs
        for i, factors in enumerate(factor_value):
            _x = define_inputs(factor_value=factors, input_vars=self.inputs_var, poly_degree=self.poly_degree)
            _y = pd.DataFrame({pred_var: define_prediction(factor_value=factors, pred_var=pred_var, decoder=self.decoder, key=f'{self.ticker}.market_price') for pred_var in self.pred_var})

            x_train.append(_x)
            y_train.append(_y)

        x = pd.concat(x_train)
        y = pd.concat(y_train)

        valid_mask = np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(y), axis=1)
        x = x[valid_mask]
        y = y[valid_mask]

        # fit the model
        coefficient = {}
        for pred_var in self.pred_var:
            if pred_var in self.model:
                model = self.model[pred_var]
            else:
                model = self.model[pred_var] = self._init_model(pred_var=pred_var, y=y[pred_var].to_numpy())

            LOGGER.info(f'fitting prediction target {pred_var}...')
            model.fit(x=x.to_numpy(), y=y[pred_var].to_numpy())
            coefficient[pred_var] = {name: value for name, value in zip(x.columns, model.coefficient)}

        report.update(coefficient='\n' + pd.DataFrame(coefficient).to_string())

        return report

    def validate(self, factor_value: pd.DataFrame, **kwargs):
        import plotly.graph_objects as go

        x_val = define_inputs(factor_value=factor_value, input_vars=self.inputs_var, poly_degree=self.poly_degree)
        y_val = pd.DataFrame({pred_var: define_prediction(factor_value=factor_value, pred_var=pred_var, decoder=self.decoder) for pred_var in self.pred_var})
        x_axis = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in x_val.index]
        candle_sticks = kwargs.get('candle_sticks')

        if candle_sticks is None:
            candle_sticks = go.Candlestick(
                name=self.ticker,
                x=x_axis,
                open=factor_value[f'{self.ticker}.open_price'],
                high=factor_value[f'{self.ticker}.high_price'],
                low=factor_value[f'{self.ticker}.low_price'],
                close=factor_value[f'{self.ticker}.close_price'],
                yaxis='y3'
            )
        else:
            candle_sticks['name'] = self.ticker
            candle_sticks['yaxis'] = 'y3'

        for pred_var in self.pred_var:
            valid_mask = np.all(np.isfinite(x_val), axis=1) & np.all(np.isfinite(y_val), axis=1)

            # setup cv
            cv = CrossValidation(model=self.model[pred_var])
            cv.x_axis = np.array(x_axis)[valid_mask]
            cv.validate(
                x_train=x_val.to_numpy(),  # have no effect
                y_train=y_val.to_numpy(),  # have no effect
                x_val=x_val.to_numpy()[valid_mask],
                y_val=y_val[pred_var].to_numpy()[valid_mask],
                skip_fitting=True
            )

            # update figure
            fig = cv.plot()
            fig.add_trace(candle_sticks)

            for level in range(self.decoder.level + 1):
                local_extreme = self.decoder.local_extremes(ticker=f'{self.ticker}.market_price', level=level)

                if not local_extreme:
                    break

                y, x, wave_flag = zip(*local_extreme)
                x = [datetime.datetime.fromtimestamp(_, tz=TIME_ZONE) for _ in x]

                trace = go.Scatter(x=x, y=y, mode='lines', name=f'decode level {level}', yaxis='y3')
                fig.add_trace(trace)

            fig.update_layout(
                yaxis3=dict(
                    title=self.ticker,
                    anchor="x",
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )

            self.cv[pred_var] = cv

        return self.cv

    def clear(self):
        self.model.clear()

    def predict(self, factor_value: dict[str, float], **kwargs) -> dict[str, float]:
        x = define_inputs(factor_value=factor_value, input_vars=self.inputs_var, poly_degree=self.poly_degree, timestamp=kwargs.get('timestamp', time.time()))
        y = {}

        for pred_var in self.pred_var:
            model = self.model[pred_var]
            y_pred, prediction_interval, resampled_deviation, *_ = model.predict(x=list(x.values()), alpha=self.alpha)
            lower_bound = y_pred + prediction_interval[0]
            upper_bound = y_pred + prediction_interval[1]

            y[pred_var] = y_pred
            y[f'{pred_var}.lower_bound'] = lower_bound
            y[f'{pred_var}.upper_bound'] = upper_bound

            if pred_var in ['pct_change', 'target_actual', 'target_smoothed']:
                kelly_proportion = kelly_bootstrap(outcomes=np.array(resampled_deviation) + y_pred, cost=self.trade_cost, max_leverage=kwargs.get('max_leverage', 2.))
                y[f'{pred_var}.kelly'] = kelly_proportion

        return y

    def predict_batch(self, factor_value: pd.DataFrame, **kwargs):
        x = define_inputs(factor_value=factor_value, input_vars=self.inputs_var, poly_degree=self.poly_degree)
        y = {}

        for pred_var in self.pred_var:
            model = self.model[pred_var]
            y_pred, prediction_interval, resampled_deviation, *_ = model.predict(x=x.to_numpy(), alpha=self.alpha)
            lower_bound = y_pred + prediction_interval[:, 0]
            upper_bound = y_pred + prediction_interval[:, 1]

            y[pred_var] = y_pred
            y[f'{pred_var}.lower_bound'] = lower_bound
            y[f'{pred_var}.upper_bound'] = upper_bound

            if pred_var in ['pct_change', 'target_actual', 'target_smoothed']:
                kelly_value = []
                for outcome, deviations in zip(y_pred, resampled_deviation):
                    kelly_proportion = kelly_bootstrap(outcomes=np.array(deviations) + outcome, cost=kwargs.get('cost', 0.0001), max_leverage=kwargs.get('max_leverage', 2.))
                    kelly_value.append(kelly_proportion)
                y[f'{pred_var}.kelly'] = np.array(kelly_value)

        return pd.DataFrame(y, index=x.index)

    @property
    def is_ready(self):
        if self.model:
            return True

        return False
