import abc

import numpy as np

from .. import Regression, LOGGER

LOGGER = LOGGER.getChild('Boosting')


class Boosting(Regression, metaclass=abc.ABCMeta):

    def resample(self, x: np.ndarray, alpha_range: list[float] = None) -> np.ndarray:
        if alpha_range is None:
            alpha_range = np.linspace(0., 1, 100)

        y_quantile = {}

        for alpha in alpha_range:
            y_pred, prediction_interval, *_ = self.predict(x=x, alpha=alpha)
            lower_bound = y_pred + prediction_interval[:, 0]
            upper_bound = y_pred + prediction_interval[:, 1]
            y_quantile[alpha / 2] = lower_bound
            y_quantile[1 - alpha / 2] = upper_bound
            y_quantile[0.5] = y_pred

        outcomes = [y[1] for y in sorted(zip(y_quantile.keys(), y_quantile.values()), key=lambda _: _[0])]
        avg_outcomes = []

        for i in range(1, len(outcomes)):
            avg_outcomes.append((outcomes[i - 1] + outcomes[i]) / 2)
        outcome_array = np.array(avg_outcomes).T

        return np.array(outcome_array)

    @property
    def ensemble(self) -> str:
        return 'boosting'


from .xgboost import *
