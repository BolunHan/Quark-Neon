import json

import forestci
import numpy as np
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from .. import LOGGER
from ..Boosting import Boosting

__all__ = ['RandomForest']


class RandomForest(Boosting):
    """
    use https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4286302/ to estimate variance
    however this relays on the normal distribution assumption
    """

    def __init__(self):
        """
        Initialize the TimeSeriesRandomForestRegressor.

        Parameters:
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int or None): The maximum depth of the tree. None means unlimited.
        - random_state (int or None): Seed for random number generation.
        """
        # Set default hyperparameters
        self.param_grid = {
            'n_estimators': [25, 50, 75, 100],
            'max_depth': [10, 20, 30],
            # 'ccp_alpha': [0.0, 0.1, 0.2, 0.5, 1.0],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['friedman_mse']
        }

        self.model = RandomForestRegressor(**{'criterion': 'friedman_mse', 'max_depth': 10, 'max_features': 'log2', 'n_estimators': 75})

        self._x_train = None
        self._y_train = None

    def fit(self, x: np.ndarray, y: np.ndarray, validation_size=0.2, optimize_params=True):
        """
        Fit the TimeSeriesRandomForestRegressor to the training data using grid search and cross-validation.

        Parameters:
        - X (array-like or pd.DataFrame): Training input data.
        - y (array-like or pd.Series): Target values.
        - validation_size (float): Proportion of the dataset to include in the validation split.

        Returns:
        None
        """
        if optimize_params:
            # Time series split for training and validation
            tscv = TimeSeriesSplit(n_splits=int(1 / validation_size))

            # Grid search with cross-validation
            grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=tscv)
            grid_search.fit(x, y)

            # Set the best hyperparameters to the model
            self.model = grid_search.best_estimator_
            LOGGER.info(f'Best model with score {grid_search.score(x, y)}, params {grid_search.best_params_}')
        else:
            self.model.fit(x, y)

        self._x_train = x
        self._y_train = y

    def predict(self, x: np.ndarray, alpha=0.05):
        """
        Make predictions using the TimeSeriesRandomForestRegressor.

        Parameters:
        - X (array-like or pd.DataFrame): Input data for prediction.
        - alpha (float): Significance level for prediction interval.

        Returns:
        - predictions (np.array): Predicted values.
        - prediction_interval (tuple): Lower and upper bounds of the prediction interval.
        """
        # Make point predictions

        x_array = np.array(x)
        single_obs = False

        # Single observation
        if x_array.ndim == 0:
            single_obs = True

        predictions = self.model.predict(x)
        variance = forestci.random_forest_error(forest=self.model, X_train=self._x_train, X_test=x)
        std = np.sqrt(variance)
        multiplier = scipy.stats.t.isf(alpha / 2, self.model.n_estimators)

        # Calculate prediction interval
        lower_bound = -std * multiplier
        upper_bound = std * multiplier

        if single_obs:
            return predictions[0], (lower_bound[0], upper_bound[0])

        prediction_interval = np.array([lower_bound, upper_bound]).T
        return predictions, prediction_interval

    def resample(self, x: np.ndarray, alpha_range: list[float] = None) -> np.ndarray:
        if alpha_range is None:
            alpha_range = np.linspace(0., 1, 100)

        y_pred = self.model.predict(x)
        variance = forestci.random_forest_error(forest=self.model, X_train=self._x_train, X_test=x)
        std = np.sqrt(variance)

        y_quantile = {}

        for alpha in alpha_range:
            multiplier = scipy.stats.t.isf(alpha / 2, self.model.n_estimators)
            lower_bound = y_pred - std * multiplier
            upper_bound = y_pred + std * multiplier
            y_quantile[alpha / 2] = lower_bound
            y_quantile[1 - alpha / 2] = upper_bound
            y_quantile[0.5] = y_pred

        outcomes = [y[1] for y in sorted(zip(y_quantile.keys(), y_quantile.values()), key=lambda _: _[0])]
        avg_outcomes = []

        for i in range(1, len(outcomes)):
            avg_outcomes.append((outcomes[i - 1] + outcomes[i]) / 2)
        outcome_array = np.array(avg_outcomes).T

        return np.array(outcome_array)

    def to_json(self, fmt='dict') -> dict | str:

        json_dict = dict(
            # model=pickle.dumps(self.model),
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

        self = cls()

        # self.model = pickle.loads(json_dict['model'])
        self._x_train = np.array(json_dict["x_train"]) if json_dict["x_train"] is not None else None
        self._y_train = np.array(json_dict["y_train"]) if json_dict["y_train"] is not None else None

        return self
