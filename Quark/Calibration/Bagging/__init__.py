import abc

from .. import Regression


class Bagging(Regression, metaclass=abc.ABCMeta):

    @property
    def ensemble(self) -> str:
        return 'bagging'
