import pandas as pd

from .. import LOGGER

LOGGER = LOGGER.getChild('Linear')


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


from .linear import *
# from .ridge import *
from Quark.Calibration.kelly import *

__all__ = ['LOGGER', 'LinearDecisionCore']
