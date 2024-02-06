import abc

import numpy as np

from .. import LOGGER, Regression

LOGGER = LOGGER.getChild('Linear')


class LinearBootstrap(Regression, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def bootstrap_standard(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Generate bootstrap samples using the standard method.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.

        Returns:
            None
        """
        ...

    @abc.abstractmethod
    def bootstrap_block(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Generate bootstrap samples using the block bootstrap method.

        Args:
            x (list or numpy.ndarray): Input features.
            y (list or numpy.ndarray): Output values.

        Returns:
            None
        """
        ...

    @property
    def ensemble(self) -> str:
        return 'bagging'


from .bootstrap import *
