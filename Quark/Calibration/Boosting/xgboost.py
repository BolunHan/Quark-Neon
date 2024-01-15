import json
import tempfile
from functools import partial

import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from . import Boosting, LOGGER
from ..Kernel import Scaler

__all__ = ['XGBoost']

LOGGER = LOGGER.getChild('xgb')


class XGBQuantile(object):
    def __init__(self, **kwargs):
        self.best_iteration: int = 0
        self.booster: xgb.Booster | None = None

        if 'alpha' in kwargs:  # assume alpha is a float from 0 to 1
            self.alpha_range = [kwargs.get('alpha')]
        elif 'alpha_range' in kwargs:  # assume alpha is a list of float from 0 to 1
            self.alpha_range = sorted(kwargs.get('alpha_range'))

        self.quantile = sorted(list(set([_ / 2 for _ in self.alpha_range] + [0.5] + [1 - _ / 2 for _ in self.alpha_range])))

    def fit(self, x: np.ndarray, y: np.ndarray, val_split: float = 0.2, max_bin: int = 256, **kwargs):
        evals_result = {}

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val_split)
        train_matrix = xgb.QuantileDMatrix(x_train, y_train, max_bin=max_bin)
        test_matrix = xgb.QuantileDMatrix(x_test, y_test, ref=train_matrix, max_bin=max_bin)

        train_params = dict(
            objective="reg:quantileerror",
            tree_method="hist",
            quantile_alpha=self.quantile,
            learning_rate=0.01,
            subsample=0.8,
            max_depth=8,
            max_bin=max_bin,
        )
        train_params.update(kwargs)

        booster = xgb.train(
            params=train_params,
            dtrain=train_matrix,
            num_boost_round=128,
            early_stopping_rounds=10,
            # The evaluation result is a weighted average across multiple quantiles.
            evals=[(train_matrix, "Train"), (test_matrix, "Test")],
            evals_result=evals_result,
            verbose_eval=False
        )

        self.booster = booster
        self.best_iteration = booster.best_iteration

        return booster

    def predict(self, x: np.ndarray) -> dict[float, np.ndarray]:
        y_pred = self.booster.inplace_predict(x, iteration_range=(0, self.best_iteration + 1))

        result = {}

        if len(self.quantile) == 1:
            result[self.quantile[0]] = y_pred
        else:
            for i in range(len(self.quantile)):
                alpha = self.quantile[i]
                y_quantile = y_pred[:, i]
                result[alpha] = y_quantile

        return result

    def to_json(self, fmt='dict') -> dict | str:

        # Create a temporary file to store the JSON representation
        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=True) as temp_file:
            json_path = temp_file.name
            self.booster.save_model(json_path)

            # Read the JSON file and parse it into a dictionary
            with open(json_path, 'r') as json_file:
                model_json = json.load(json_file)

        json_dict = dict(
            best_iteration=self.best_iteration,
            alpha_range=self.alpha_range,
            quantile=self.quantile,
            model=model_json
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
            alpha_range=json_dict['alpha_range']
        )

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+', delete=True) as temp_file:
            json_path = temp_file.name
            json.dump(json_dict, temp_file)
            model = xgb.Booster(model_file=json_path)

        self.booster = model
        self.quantile = json_dict['quantile']
        self.best_iteration = json_dict['best_iteration']

        return self


class XGBQuantileCustomLoss(object):
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
    def __init__(self, alpha: float = 0.05):
        super().__init__()

        self.alpha: float = alpha
        self.model = XGBQuantile(alpha=self.alpha)
        self.model_cache = {}

        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self.val_split = 0.2

    def fit(self, x: list[float] | np.ndarray, y: list[float] | np.ndarray, **kwargs) -> None:
        self.model.fit(x=x, y=y, val_split=self.val_split)

        self._x_train = x
        self._y_train = y
        self.model_cache.clear()

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

        if alpha in self.model.alpha_range:
            y_dict = self.model.predict(x_array)
        else:
            y_dict = self.predict_quantile(x=x_array, alpha=alpha)

        y_lower = y_dict[alpha / 2]
        y_med = y_dict[0.5]
        y_upper = y_dict[1 - alpha / 2]

        if single_obs:
            return y_med[0], (y_lower[0], y_upper[0])

        return y_med, np.array([y_lower, y_upper]).T

    def predict_quantile(self, x: np.ndarray, alpha: float) -> dict[float, np.ndarray]:

        if alpha in self.model_cache:
            model = self.model_cache[alpha]
        else:
            LOGGER.debug(f'prediction interval of alpha {alpha:.4f} is not cached!')
            model = XGBQuantile(alpha=alpha)
            model.fit(x=self._x_train, y=self._y_train, val_split=self.val_split)
            self.model_cache[alpha] = model

        y_dict = model.predict(x=x)
        return y_dict

    def to_json(self, fmt='dict') -> dict | str:

        json_dict = dict(
            alpha=self.alpha,
            val_split=self.val_split,
            model=self.model.to_json(fmt='dict'),
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
            alpha=json_dict['alpha']
        )

        self.val_split = json_dict['val_split']
        self.model = XGBQuantile.from_json(json_dict['model'])
        self._x_train = np.array(json_dict["x_train"]) if json_dict["x_train"] is not None else None
        self._y_train = np.array(json_dict["y_train"]) if json_dict["y_train"] is not None else None

        return self

    @property
    def is_fitted(self) -> bool:
        if self._x_train is not None and self._y_train is not None:
            return True

        return False
