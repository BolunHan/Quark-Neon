import json

import numpy as np
import pandas as pd

from . import Scaler
from .linear import LinearRegressionCore, LogLinearCore
from ...Base import GlobalStatics

TIME_ZONE = GlobalStatics.TIME_ZONE


class RidgeLinearCore(LinearRegressionCore, Scaler):
    def __init__(self, ticker: str, **kwargs):
        self.alpha = kwargs.get('ridge_alpha', 1)

        super().__init__(ticker=ticker, **kwargs)
        Scaler.__init__(self)

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

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, input_cols: list[str] = None):
        self.standardization_scaler(x)
        x = self.transform(x)
        return super().fit(x=x, y=y, input_cols=input_cols)

    def _fit(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float64)
        y = y.astype(np.float64)

        x_transpose = x.T
        xtx = np.dot(x_transpose, x)
        xty = np.dot(x_transpose, y)
        identity_matrix = np.identity(x.shape[1])  # or np.identity(len(xtx))
        regularization_term = self.alpha * identity_matrix
        xtx_plus_reg = xtx + regularization_term
        xtx_inv = np.linalg.inv(xtx_plus_reg)
        coefficients = np.dot(xtx_inv, xty)
        residuals = y - np.dot(x, coefficients)
        mse = np.mean(residuals ** 2, axis=0)

        return coefficients, mse

    def _pred(self, x: np.ndarray | pd.DataFrame | dict[str, float]) -> np.ndarray | dict[str, float]:
        x = self.transform(x)
        return super()._pred(x=x)


class RidgeDecodingCore(LogLinearCore, RidgeLinearCore):
    def __init__(self, ticker: str, **kwargs):
        super(LogLinearCore, self).__init__(ticker=ticker, **kwargs)
        super(RidgeLinearCore, self).__init__(ticker=ticker, **kwargs)

        # the pred var is overridden by the RidgeLinearCore.__init__
        self.pred_var = ['up_smoothed', 'down_smoothed']

    def to_json(self, fmt='dict') -> dict | str:
        return super(RidgeLinearCore, self).to_json(fmt=fmt)

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        return super(RidgeLinearCore, cls).from_json(json_str=json_str)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame, input_cols: list[str] = None):
        return super(RidgeLinearCore, self).fit(x=x, y=y, input_cols=input_cols)

    def _fit(self, x: np.ndarray, y: np.ndarray):
        return super(RidgeLinearCore, self)._fit(x=x, y=y)

    def _pred(self, x: np.ndarray | pd.DataFrame | dict[str, float]) -> np.ndarray | dict[str, float]:
        return super(RidgeLinearCore, self)._pred(x=x)


__all__ = ['RidgeLinearCore', 'RidgeDecodingCore']
