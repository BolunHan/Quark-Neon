import json
from functools import partial

import numpy as np
from xgboost.sklearn import XGBRegressor

from . import Boosting, LOGGER
from ..Kernel import Scaler

__all__ = ['XGBoost']

LOGGER = LOGGER.getChild('xgb')


class XGBQuantile(object):
    """
    see https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b for detailed explanation
    codes refactored from https://colab.research.google.com/github/benoitdescamps/benoit-descamps-blogs/blob/master/notebooks/quantile_xgb/xgboost_quantile_regression.ipynb
    """

    def __init__(
            self,
            # base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
            # max_depth=3, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear',
            # random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, subsample=1,
            **kwargs
    ):
        self.model = XGBRegressor(
            # base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
            # max_depth=3, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear',
            # random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, subsample=1,
            **kwargs
        )

        self.x_scaler = Scaler()
        self.y_scaler = Scaler()

        self.alpha = None
        self.delta = None
        self.threshold = None
        self.variance = None

    def fit(self, x, y, quantile_alpha: float = None, quantile_delta=1.0, quantile_threshold=5.0, quantile_variance=3.2, **kwargs):
        if not self.x_scaler.is_ready:
            self.x_scaler.fit(x)

        if not self.y_scaler.is_ready:
            self.y_scaler.fit(y)

        x = self.x_scaler.transform(x)
        y = self.y_scaler.transform(y)

        # override the objective and score if the model regression target is quantile
        if quantile_alpha is not None:
            self.model.set_params(
                objective=partial(
                    self.quantile_loss,
                    alpha=quantile_alpha,
                    delta=quantile_delta,
                    threshold=quantile_threshold,
                    variance=quantile_variance
                )
            )
            self.model.score = partial(self.score, alpha=quantile_alpha)

            self.alpha = quantile_alpha
            self.delta = quantile_delta
            self.threshold = quantile_threshold
            self.variance = quantile_variance

        self.model.fit(x, y, **kwargs)
        return self

    def predict(self, x: np.ndarray):
        x = self.x_scaler.transform(x)

        y = self.model.predict(x)

        y = self.y_scaler.reverse_transform(y)
        return y

    def score(self, x, y, alpha: float):
        y_pred = self.predict(x)
        score = self.quantile_score(y, y_pred, alpha)
        score = 1. / score
        return score

    @classmethod
    def quantile_loss(cls, y_true, y_pred, alpha, delta, threshold, variance):
        # note: check the y is not standardized before this fitting operation
        x = y_true - y_pred
        grad, hess = cls.original_quantile_loss(y_true=y_true, y_pred=y_pred, alpha=alpha, delta=delta)
        # randomly +- a variance, note, the y is not standardized before this operation
        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (2 * np.random.randint(2, size=len(y_true)) - 1.0) * variance
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    @classmethod
    def original_quantile_loss(cls, y_true, y_pred, alpha, delta):
        x = y_true - y_pred
        grad = (x < (alpha - 1.0) * delta) * (1.0 - alpha) - ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta - alpha * (x > alpha * delta)
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
        return grad, hess

    @classmethod
    def quantile_score(cls, y_true, y_pred, alpha):
        score = cls.quantile_cost(x=y_true - y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l) + np.sum(gradient[i:]) / (np.sum(hessian[i:]) + l) - np.sum(gradient) / (np.sum(hessian) + l))

        return np.array(split_gain)


class XGBoost(Boosting):
    def __init__(self, alpha: float = 0.05, lower_bound_params: dict[str, float] = None, upper_bound_params: dict[str, float] = None, **kwargs):
        super().__init__()

        self.alpha: float = alpha
        self.lower_bound_params = dict(quantile_delta=1.0, quantile_threshold=5.0, quantile_variance=3.2) if lower_bound_params is None else lower_bound_params
        self.upper_bound_params = dict(quantile_delta=1.0, quantile_threshold=6.0, quantile_variance=4.2) if upper_bound_params is None else upper_bound_params

        self.xgb_params = {}
        self.xgb_lower_params = {}
        self.xgb_upper_params = {}

        self.xgb_params.update(kwargs)

        if 'lower_model' in self.xgb_params:
            self.xgb_lower_params.update(self.xgb_params.pop('lower_model'))
        else:
            self.xgb_lower_params = self.xgb_params

        if 'upper_model' in self.xgb_params:
            self.xgb_upper_params.update(self.xgb_params.pop('upper_model'))
        else:
            self.xgb_upper_params = self.xgb_params

        self.quantile_model_lower = XGBQuantile(**self.xgb_lower_params)
        self.quantile_model_upper = XGBQuantile(**self.xgb_upper_params)
        self.model = XGBQuantile(**self.xgb_params)

        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x: list[float] | np.ndarray, y: list[float] | np.ndarray, **kwargs) -> None:
        self.model.fit(x=x, y=y, **kwargs)

        self.quantile_model_lower.fit(
            x=x,
            y=y,
            quantile_alpha=self.alpha / 2,
            quantile_delta=self.lower_bound_params['quantile_delta'],
            quantile_threshold=self.lower_bound_params['quantile_threshold'],
            quantile_variance=self.lower_bound_params['quantile_variance'],
            **kwargs
        )

        self.quantile_model_upper.fit(
            x=x,
            y=y,
            quantile_alpha=1 - self.alpha / 2,
            quantile_delta=self.upper_bound_params['quantile_delta'],
            quantile_threshold=self.upper_bound_params['quantile_threshold'],
            quantile_variance=self.upper_bound_params['quantile_variance'],
            **kwargs
        )

        self._x_train = x
        self._y_train = y

    def predict(self, x: list | np.ndarray, alpha=0.05):
        if not self.is_fitted:
            raise ValueError(f'Model {self.__class__} must be fitted before prediction!')

        x_array = np.array(x)
        single_obs = False

        # Single observation
        if x_array.ndim == 0:
            single_obs = True

        if single_obs:
            x_array = x_array.reshape(1, -1)

        y_pred = self.model.predict(x_array)

        lower_bound_params = dict(quantile_alpha=alpha / 2)
        upper_bound_params = dict(quantile_alpha=1.0 - alpha / 2)
        lower_bound_params.update(self.lower_bound_params)
        upper_bound_params.update(self.upper_bound_params)

        y_lower = self.predict_quantile(x_array, **lower_bound_params)
        y_upper = self.predict_quantile(x_array, **upper_bound_params)

        y_lower -= y_pred
        y_upper -= y_pred

        if single_obs:
            return y_pred[0], (y_lower[0], y_upper[0])

        return y_pred, np.array([y_lower, y_upper]).T

    def predict_quantile(self, x: np.ndarray, quantile_alpha: float, quantile_delta: float, quantile_threshold: float, quantile_variance: float) -> np.ndarray:
        if quantile_alpha < 0.5:
            model = self.quantile_model_lower
        else:
            model = self.quantile_model_upper

        if model.alpha != quantile_alpha:
            LOGGER.debug(f'Cache is not hit! requested model with alpha {quantile_alpha:.4f}, cached model alpha {model.alpha:.4f}, model will be fitted online.')

            if self._x_train is None or self._y_train is None:
                raise ValueError(f'Model with alpha {quantile_alpha} is not cached, The Model {self.__class__} must be fitted before prediction!')

            model.fit(
                x=self._x_train,
                y=self._y_train,
                quantile_alpha=quantile_alpha,
                quantile_delta=quantile_delta,
                quantile_threshold=quantile_threshold,
                quantile_variance=quantile_variance
            )

        return model.predict(x)

    def to_json(self, fmt='dict') -> dict | str:

        json_dict = dict(
            alpha=self.alpha,
            lower_bound_params=self.lower_bound_params,
            upper_bound_params=self.upper_bound_params,
            xgb_params=self.xgb_params,
            xgb_lower_params=self.xgb_lower_params,
            xgb_upper_params=self.xgb_upper_params,
            x_train=self._x_train.tolist() if self._x_train is not None else None,
            y_train=self._y_train.tolist() if self._y_train is not None else None,
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
            alpha=json_dict['alpha'],
            lower_bound_params=json_dict['lower_bound_params'],
            upper_bound_params=json_dict['upper_bound_params'],
            xgb_lower_params=json_dict['lower_model'],
            xgb_upper_params=json_dict['upper_model'],
            **json_dict['xgb_params']
        )

        self._x_train = np.array(json_dict["x_train"]) if json_dict["x_train"] is not None else None
        self._y_train = np.array(json_dict["y_train"]) if json_dict["y_train"] is not None else None

        if self._x_train is not None and self._y_train is not None:
            self.fit(x=self._x_train, y=self._y_train)

        return self

    @property
    def is_fitted(self) -> bool:
        if self._x_train is not None and self._y_train is not None:
            return True

        return False
