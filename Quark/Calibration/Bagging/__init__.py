import abc

from ..Linear import Bootstrap


class Bagging(Bootstrap, metaclass=abc.ABCMeta):

    @property
    def ensemble(self) -> str:
        return 'bagging'
