import abc
import json

import numpy as np

__all__ = ['Transformer', 'LogTransformer', 'SigmoidTransformer', 'BinaryTransformer']


class Transformer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def inverse_transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def mask(self, y: list[float] | np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def to_json(self, fmt='dict') -> dict | str:
        ...

    @classmethod
    def from_json(cls, json_str: str | bytes | dict):
        if isinstance(json_str, (str, bytes)):
            json_dict = json.loads(json_str)
        elif isinstance(json_str, dict):
            json_dict = json_str
        else:
            raise TypeError(f'{cls.__name__} can not load from json {json_str}')

        type_str = json_dict['type']

        if type_str == 'LogTransformer':
            return LogTransformer.from_json(json_str)
        elif type_str == 'SigmoidTransformer':
            return SigmoidTransformer.from_json(json_str)
        elif type_str == 'BinaryTransformer':
            return BinaryTransformer.from_json(json_str)
        else:
            raise NotImplementedError(f'Invalid {cls.__name__} type {type_str}.')

    def transform_safe(self, y: np.ndarray):
        mask = self.mask(y)
        _y = y[mask]
        _y = self.transform(_y)
        return _y, mask


class LogTransformer(Transformer):
    def __init__(self, is_negative: bool = False, boundary: float = 0.):
        self.is_negative = is_negative
        self.boundary = boundary

    def transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        """
        transform from [boundary, inf) -> (-inf, inf)
        Args:
            y:

        Returns:

        """
        if not self.is_negative:
            _y = np.log(y - self.boundary)
        else:
            _y = np.log(self.boundary - y)

        return _y

    def inverse_transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        if not self.is_negative:
            _y = np.exp(y) + self.boundary
        else:
            _y = self.boundary - np.exp(y)

        return _y

    def mask(self, y: list[float] | np.ndarray) -> np.ndarray:
        if not self.is_negative:
            mask = np.greater(y, self.boundary)
        else:
            mask = np.less(y, self.boundary)

        return mask

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            type=f'{self.__class__.__name__}',
            is_negative=self.is_negative,
            boundary=self.boundary
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
            is_negative=json_dict['is_negative'],
            boundary=json_dict['boundary'],
        )

        return self


class SigmoidTransformer(Transformer):
    def __init__(self, upper_bound: float = 1., lower_bound: float = 0.):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        """
        transform from [lower_bound, upper_bound) -> (-inf, inf)

        Args:
            y:

        Returns:

        """
        scale = (self.upper_bound - self.lower_bound)
        center = self.lower_bound

        _y = -np.log(scale / (y - center) - 1)

        return _y

    def inverse_transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        scale = (self.upper_bound - self.lower_bound)
        center = self.lower_bound

        _y = scale / (1 + np.exp(-y)) + center
        return _y

    def mask(self, y: list[float] | np.ndarray) -> np.ndarray:
        mask = np.less(y, self.upper_bound) & np.greater(y, self.lower_bound)
        return mask

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            type=f'{self.__class__.__name__}',
            upper_bound=self.upper_bound,
            lower_bound=self.lower_bound
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
            upper_bound=json_dict['upper_bound'],
            lower_bound=json_dict['lower_bound'],
        )

        return self


class BinaryTransformer(Transformer):
    def __init__(self, center: float = None, scale: float = 1):
        self.center = center
        self.scale = scale

    def transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        """
        transform from (-inf, inf) -> {0, 1}

        Args:
            y:

        Returns:

        """
        if self.center is None:
            self.center = np.nanmedian(y)

        _y = np.array((np.sign(y - self.center) + 1) / 2).astype(int)
        return _y

    def inverse_transform(self, y: list[float] | np.ndarray) -> np.ndarray:
        _y = (y * 2 - 1) * self.scale + self.center
        return _y

    def mask(self, y: list[float] | np.ndarray) -> np.ndarray:
        if self.center is None:
            self.center = np.nanmedian(y)

        mask = np.isfinite(y) & (y != self.center)
        return mask

    def to_json(self, fmt='dict') -> dict | str:
        json_dict = dict(
            type=f'{self.__class__.__name__}',
            center=self.center,
            scale=self.scale
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
            center=json_dict['center'],
            scale=json_dict['scale']
        )

        return self
