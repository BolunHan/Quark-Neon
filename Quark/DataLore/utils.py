from typing import Iterable

import numpy as np
import pandas as pd

from . import LOGGER
from ..Calibration.Kernel import poly_kernel
from ..Calibration.dummies import is_market_session, session_dummies
from ..Factor.decoder import RecursiveDecoder
from ..Factor.future import fix_prediction_target, wavelet_prediction_target


class Scaler(object):
    def __init__(self):
        self.scaler: pd.DataFrame | None = None

    def standardization_scaler(self, x: pd.DataFrame):
        scaler = pd.DataFrame(index=['mean', 'std'], columns=x.columns)

        for col in x.columns:
            if col == 'Bias':
                scaler.loc['mean', col] = 0
                scaler.loc['std', col] = 1
            else:
                valid_values = x[col][np.isfinite(x[col])]
                scaler.loc['mean', col] = np.mean(valid_values)
                scaler.loc['std', col] = np.std(valid_values)

        self.scaler = scaler
        return scaler

    def transform(self, x: pd.DataFrame | dict[str, float]) -> pd.DataFrame | dict[str, float]:
        if self.scaler is None:
            raise ValueError('scaler not initialized!')

        if isinstance(x, pd.DataFrame):
            x = (x - self.scaler.loc['mean']) / self.scaler.loc['std']
        elif isinstance(x, dict):
            for var_name in x:

                if var_name not in self.scaler.columns:
                    # LOGGER.warning(f'{var_name} is not in scaler')
                    continue

                x[var_name] = (x[var_name] - self.scaler.at['mean', var_name]) / self.scaler.at['std', var_name]
        else:
            raise TypeError(f'Invalid x type {type(x)}, expect dict or pd.DataFrame')

        return x


def define_inputs(factor_value: pd.DataFrame | dict[str, float | int], input_vars: Iterable[str], poly_degree: int = 1, timestamp: float | list[float] = None) -> pd.DataFrame:
    """
    Defines input features for regression analysis.

    Args:
        timestamp: pass in this parameter to override timestamp
        poly_degree: degree for poly feature extension
        input_vars: a list or iterable of selected input variables
        factor_value (pd.DataFrame): DataFrame containing factors.
    """
    if timestamp is None:
        session_dummies(timestamp=factor_value.index, inplace=factor_value)
    else:
        session_dummies(timestamp=timestamp, inplace=factor_value)

    # default features
    features = {'Dummies.IsOpening', 'Dummies.IsClosing'}
    features.update(input_vars)

    # generate copies
    if isinstance(factor_value, pd.DataFrame):
        feature_original = factor_value[list(features)].copy()
        x_matrix = factor_value.loc[:, list(features)].copy()
    elif isinstance(factor_value, dict):
        feature_original = {_: factor_value.get(_, np.nan) for _ in features}
        x_matrix = {_: factor_value.get(_, np.nan) for _ in features}
    else:
        raise TypeError(f'Invalid factor value, expect dict[str, float] or pd.DataFrame, got {type(factor_value)}.')

    # extend features
    for i in range(1, poly_degree):
        additional_feature = poly_kernel(feature_original, degree=i + 1)
        if isinstance(x_matrix, pd.DataFrame):
            x_matrix = pd.concat([x_matrix, pd.DataFrame(additional_feature, index=x_matrix.index)], axis=1)
        else:
            for _ in additional_feature:
                x_matrix[_] = additional_feature[_]

    # check multicolinearity
    if isinstance(x_matrix, pd.DataFrame):
        invalid_factor: pd.DataFrame = x_matrix.loc[:, x_matrix.nunique() == 1]
        if not invalid_factor.empty:
            LOGGER.error(f'Invalid factor {invalid_factor.columns}, add epsilon to avoid multicolinearity')
            for name in invalid_factor.columns:
                x_matrix[name] = 1 + np.random.normal(scale=0.1, size=len(x_matrix))

    # add bias
    x_matrix['bias'] = 1.

    return x_matrix


def define_prediction(factor_value: pd.DataFrame, pred_var: str, decoder: RecursiveDecoder, key: str = 'SyntheticIndex.market_price') -> pd.Series:
    """
    Defines the prediction target for regression analysis.

    Args:
        key: name of the column of the market price
        decoder: recursive decoder for decoding market movements
        pred_var: the prediction target
        factor_value (pd.DataFrame): DataFrame containing factors.
    """
    if pred_var == 'pct_change':
        fix_prediction_target(
            factors=factor_value,
            key=key,
            session_filter=is_market_session,
            inplace=True,
            pred_length=15 * 60
        )
    elif pred_var in ['up_actual', 'down_actual', 'target_actual',
                      'up_smoothed', 'down_smoothed', 'target_smoothed',
                      'state']:
        wavelet_prediction_target(
            factors=factor_value,
            key=key,
            session_filter=is_market_session,
            inplace=True,
            decoder=decoder,
            decode_level=decoder.level
        )
    else:
        raise NotImplementedError(f'Invalid prediction target {pred_var}.')

    y = factor_value[pred_var]
    return y


__all__ = ['define_inputs', 'define_prediction']
