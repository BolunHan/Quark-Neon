import abc
import enum
import json
import os
import pickle
import platform
import uuid
from collections import deque
from collections.abc import Iterable
from typing import Any, Self

import numpy as np

from .. import LOGGER

LOGGER = LOGGER.getChild('MemoryCore')


class MemoryBuffer(object):
    def __init__(self, /, name: str, buffer: memoryview):
        self.name = name
        self.buffer = buffer

    def __bytes__(self):
        return self.to_bytes()

    def __int__(self):
        return self.to_int()

    def __float__(self):
        return self.to_float()

    def __str__(self):
        return self.to_str()

    def __bool__(self):
        return True

    def to_bytes(self):
        return bytes(self.buffer)

    def to_int(self):
        return int.from_bytes(self.buffer)

    def to_float(self):
        return float(np.frombuffer(self.buffer)[0])

    def to_str(self, encoding='utf8'):
        return self.buffer.tobytes().decode(encoding)

    def to_array(self, shape: tuple[int, ...] = None, dtype: type = None) -> np.ndarray:
        if shape is None:
            shape = (-1,)

        return np.ndarray(shape=shape, dtype=dtype, buffer=self.buffer)

    def unpickle(self):
        return pickle.loads(self.buffer)

    @property
    def size(self):
        return len(self.buffer)


class ByteConvertor(object):
    def __init__(self, encoding='utf-8', seperator=b'\x07'):
        self.encoding = encoding
        self.seperator: bytes = seperator
        self.seperator_decoded = self.seperator.decode(encoding)

    def str_to_bytes(self, data: str) -> bytes:
        if self.seperator_decoded in data:
            LOGGER.error(f'seperator {self.seperator} in the data string, this might cause the memory buffer interpreted as a str_vector.')
        bytes_data = data.encode(self.encoding)
        return bytes_data

    def bytes_to_str(self, data: bytes | memoryview | MemoryBuffer) -> str:
        return bytes(data).decode(self.encoding)

    def str_vector_to_bytes(self, data: Iterable[str]) -> bytes:
        bytes_data = self.seperator.join([_.encode(self.encoding) for _ in data])
        return bytes_data

    def bytes_to_str_vector(self, data: bytes | memoryview | MemoryBuffer) -> list[str]:
        return [_.decode(self.encoding) for _ in bytes(data).split(self.seperator)]

    @classmethod
    def int_to_bytes(cls, data: int, int_size: int = 8) -> bytes:
        return data.to_bytes(length=int_size)

    @classmethod
    def bytes_to_int(cls, data: bytes | memoryview | MemoryBuffer) -> int:
        return int.from_bytes(bytes(data))

    @classmethod
    def float_to_bytes(cls, data: float) -> bytes:
        return np.double(data).tobytes()

    @classmethod
    def bytes_to_float(cls, data: bytes | memoryview | MemoryBuffer) -> float:
        return float(np.frombuffer(bytes(data))[0])

    @classmethod
    def array_to_bytes(cls, data: np.ndarray) -> bytes:
        return data.tobytes()

    @classmethod
    def bytes_to_array(cls, data: bytes | memoryview | MemoryBuffer, shape: tuple[int, ...] = None, dtype: type = None) -> np.ndarray:
        if shape is None:
            return np.frombuffer(bytes(data), dtype=dtype)

        return np.ndarray(shape=shape, dtype=dtype, buffer=bytes(data))

    @classmethod
    def object_to_bytes(cls, data: Any) -> bytes:
        return pickle.dumps(data)

    @classmethod
    def bytes_to_object(cls, data: bytes | memoryview | MemoryBuffer) -> Any:
        return pickle.loads(bytes(data))


class MemoryManager(object, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.convertor = ByteConvertor(
            encoding=kwargs.get('encoding', 'utf-8'),
            seperator=kwargs.get('seperator', b'\x07')
        )

    @abc.abstractmethod
    def init_buffer(self, name: str, buffer_size: int = None, init_value: bytes = None) -> MemoryBuffer:
        ...

    @abc.abstractmethod
    def get_buffer(self, name: str) -> MemoryBuffer:
        ...

    @abc.abstractmethod
    def set_buffer(self, name: str, value: bytes):
        ...

    @abc.abstractmethod
    def close_buffer(self, name: str) -> int:
        ...

    @abc.abstractmethod
    def unlink_buffer(self, name: str) -> int:
        ...

    def get_bytes(self, name: str) -> bytes:
        buffer = self.get_buffer(name=name)
        return bytes(buffer)

    def set_int(self, name: str, value: int, init_ok: bool = True):
        byte_data = self.convertor.int_to_bytes(data=value)

        if init_ok:
            res = self.init_buffer(name=name, init_value=byte_data, buffer_size=len(byte_data))
        else:
            res = self.set_buffer(name=name, value=byte_data)

        return res

    def get_int(self, name: str) -> int:
        buffer = self.get_buffer(name=name)
        return self.convertor.bytes_to_int(data=buffer)

    def set_double(self, name: str, value: float, init_ok: bool = True) -> int:
        byte_data = self.convertor.float_to_bytes(data=value)

        if init_ok:
            res = self.init_buffer(name=name, init_value=byte_data, buffer_size=len(byte_data))
        else:
            res = self.set_buffer(name=name, value=byte_data)

        return res

    def get_double(self, name: str) -> float | None:
        buffer = self.get_buffer(name=name)
        return self.convertor.bytes_to_float(data=buffer)

    def set_vector(self, name: str, vector: Iterable[float | int | bool] | np.ndarray, init_ok: bool = True) -> int:
        byte_data = self.convertor.array_to_bytes(data=np.array(vector))

        if init_ok:
            res = self.init_buffer(name=name, init_value=byte_data, buffer_size=len(byte_data))
        else:
            res = self.set_buffer(name=name, value=byte_data)

        return res

    def get_vector(self, name: str, target: list = None) -> list[float]:
        buffer = self.get_buffer(name=name)
        vector = self.convertor.bytes_to_array(shape=(-1,), data=buffer).tolist()

        if target is None:
            return vector

        target.clear()
        target.extend(vector)
        return target

    def set_str_vector(self, name: str, vector: list[str], init_ok: bool = True) -> int:
        byte_data = self.convertor.str_vector_to_bytes(data=vector)

        if init_ok:
            res = self.init_buffer(name=name, init_value=byte_data, buffer_size=len(byte_data))
        else:
            res = self.set_buffer(name=name, value=byte_data)

        return res

    def get_str_vector(self, name: str, target: list = None) -> list[str]:
        buffer = self.get_buffer(name=name)
        vector = self.convertor.bytes_to_str_vector(data=buffer)

        if target is None:
            return vector

        target.clear()
        target.extend(vector)
        return target

    def set_named_vector(self, name: str, obj: dict[str, float], init_ok: bool = True) -> None:
        """
        sync the dict[str, float]
        """
        keys_name = f'{name}.keys'
        values_name = f'{name}.values'

        self.set_str_vector(name=keys_name, vector=list(obj.keys()), init_ok=init_ok)
        self.set_vector(name=values_name, vector=list(obj.values()), init_ok=init_ok)

    def get_name_vector(self, name: str, target: dict = None) -> dict[str, float]:
        keys_name = f'{name}.keys'
        values_name = f'{name}.values'

        keys = self.get_str_vector(name=keys_name)
        values = self.get_vector(name=values_name)
        shm_dict = dict(zip(keys, values))

        if target is None:
            return shm_dict
        else:
            target.clear()
            target.update(shm_dict)
            return target

    def sync(self, name: str, obj: Any, init_ok: bool = True) -> int:
        byte_data = self.convertor.object_to_bytes(data=obj)

        if init_ok:
            res = self.init_buffer(name=name, init_value=byte_data, buffer_size=len(byte_data))
        else:
            res = self.set_buffer(name=name, value=byte_data)

        return res

    def get(self, name: str) -> Any:
        buffer = self.get_buffer(name=name)
        obj = self.convertor.bytes_to_object(data=buffer)
        return obj

    @property
    def encoding(self):
        return self.convertor.encoding


class SyncTypes(enum.StrEnum):
    Vector = 'Vector'
    NamedVector = 'NamedVector'
    Deque = 'Deque'
    IntValue = 'IntValue'
    FloatValue = 'FloatValue'


class SyncTemplate(object, metaclass=abc.ABCMeta):
    def __init__(self, manager: MemoryManager | None, name: str, **kwargs):
        self._name = name
        self._manager = manager

        if 'ver_shm' in kwargs:
            self._ver_shm = kwargs['ver_shm']
        else:
            self._ver_shm = manager.init_buffer(name=f'{name}.ver_code', init_value=b'\x00' * 16)

        self._ver_local = kwargs.get('ver_local', uuid.uuid4().bytes)
        self.desync_counter = kwargs.get('desync_counter', 0)
        self._buffer_attribute: list[str] = ['_ver_shm']

    def _register_buffer_attribute(self, attribute_name: str):
        if attribute_name in self._buffer_attribute:
            LOGGER.info(f'Attribute {attribute_name} already registered!')
            return

        self._buffer_attribute.append(attribute_name)

    @abc.abstractmethod
    def to_json(self, fmt='str'):
        ...

    @classmethod
    def from_json(cls, manager: MemoryManager, json_message: str | bytes | bytearray | dict, dummy: bool = False):
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype: str = json_dict['dtype']

        if getattr(manager, 'dummy', dummy):
            match dtype:
                case SyncTypes.NamedVector.value:
                    self = NamedVectorDummy.from_json(json_message=json_dict)
                case SyncTypes.Vector.value:
                    self = VectorDummy.from_json(json_message=json_dict)
                case SyncTypes.Deque.value:
                    self = DequeDummy.from_json(json_message=json_dict)
                case SyncTypes.IntValue.value | SyncTypes.FloatValue.value:
                    self = ValueDummy.from_json(json_message=json_dict)
                case _:
                    raise ValueError(f'Invalid dtype {dtype}.')
        else:
            match dtype:
                case SyncTypes.NamedVector.value:
                    self = NamedVector.from_json(manager=manager, json_message=json_dict)
                case SyncTypes.Vector.value:
                    self = Vector.from_json(manager=manager, json_message=json_dict)
                case SyncTypes.Deque.value:
                    self = Deque.from_json(manager=manager, json_message=json_dict)
                case SyncTypes.IntValue.value:
                    self = IntValue.from_json(manager=manager, json_message=json_dict)
                case SyncTypes.FloatValue.value:
                    self = FloatValue.from_json(manager=manager, json_message=json_dict)
                case _:
                    raise ValueError(f'Invalid dtype {dtype}.')

        return self

    @abc.abstractmethod
    def to_shm(self, override: bool = False):
        ...

    @abc.abstractmethod
    def from_shm(self, override: bool = False):
        ...

    def close(self):
        for attribute_name in self._buffer_attribute:
            buffer = getattr(self, attribute_name, None)
            if buffer is None:
                continue

            if not isinstance(buffer, MemoryBuffer):
                raise TypeError(f'Invalid buffer {attribute_name}. Expected MemoryBuffer, got {type(buffer)}.')

            self._manager.close_buffer(name=buffer.name)
            setattr(self, attribute_name, None)

        self.clear(silent=True)

    def unlink(self):
        for attribute_name in self._buffer_attribute:
            buffer = getattr(self, attribute_name, None)
            if buffer is None:
                continue

            if not isinstance(buffer, MemoryBuffer):
                raise TypeError(f'Invalid buffer {attribute_name}. Expected MemoryBuffer, got {type(buffer)}.')

            self._manager.close_buffer(name=buffer.name)
            self._manager.unlink_buffer(name=buffer.name)
            setattr(self, attribute_name, None)

        self.clear(silent=True)

    def new_ver(self, desync_warning_limit: int = 100):
        self._ver_local: bytes = uuid.uuid4().bytes
        self.desync_counter += 1
        if desync_warning_limit is not None and self.desync_counter >= desync_warning_limit > 0:
            LOGGER.error(f'{self} de-synchronize too many times, check the code!')

    def set_ver(self):
        self._ver_shm.buffer[:16] = self._ver_local

    def get_ver(self):
        self._ver_local = bytes(self._ver_shm)

    @abc.abstractmethod
    def clear(self, silent: bool = False):
        ...

    @property
    def name(self):
        return self._name

    @property
    def encoding(self):
        if self._manager is None:
            return None
        else:
            return self._manager.encoding

    @property
    def ver_shm(self) -> bytes:
        if self._ver_shm is None:
            return b''

        return bytes(self._ver_shm)

    @property
    def ver_local(self) -> bytes:
        return self._ver_local

    @property
    def is_sync(self) -> bool:
        """
        A bool indicating if the data is connected to the shared memory
        Returns:

        """
        if self.ver_local == self.ver_shm:
            return True

        return False

    @property
    def empty_shm(self) -> bool:
        """
        if ver_shm is all null, for sure the shm is not initialized.
        Returns:

        """
        if self.ver_shm == b'\x00' * 16 or self.ver_shm == b'':
            return True

        return False


class NamedVectorDummy(SyncTemplate, dict):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(name=name, manager=None, ver_shm=None)
        dict.__init__(self=self, *args, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.removesuffix('Dummy'),
            name=self._name,
            data=dict(self),
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            name=json_dict['name'],
        )

        return self

    def to_shm(self, override: bool = False):
        pass

    def from_shm(self, override: bool = False):
        pass

    def unlink(self):
        self.clear()

    def clear(self, silent: bool = False):
        dict.clear(self)

    @property
    def is_sync(self) -> bool:
        return False

    @is_sync.setter
    def is_sync(self, is_sync: bool):
        raise AttributeError(f'Can not set property is_sync in {self.__class__.__name__}.')


class VectorDummy(SyncTemplate, list):
    def __init__(self, *args, name: str, **kwargs):
        super().__init__(name=name, manager=None, ver_shm=None)
        list.__init__(self=self, *args, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.removesuffix('Dummy'),
            name=self._name,
            data=list(self)
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            name=json_dict['name'],
        )

        self.extend(json_dict['data'])

        return self

    def to_shm(self, override: bool = False):
        pass

    def from_shm(self, override: bool = False):
        pass

    def unlink(self):
        self.clear()

    def clear(self, silent: bool = False):
        list.clear(self)

    @property
    def is_sync(self) -> bool:
        return False


class DequeDummy(SyncTemplate, deque):
    def __init__(self, *args, name: str, maxlen: int = None, **kwargs):
        super().__init__(name=name, manager=None, ver_shm=None)
        deque.__init__(self=self, *args, maxlen=maxlen, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.removesuffix('Dummy'),
            name=self._name,
            maxlen=self.maxlen,
            data=list(self)
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            name=json_dict['name'],
            maxlen=json_dict['maxlen']
        )

        return self

    def to_shm(self, override: bool = False):
        pass

    def from_shm(self, override: bool = False):
        pass

    def unlink(self):
        self.clear()

    def clear(self, silent: bool = False):
        deque.clear(self)

    @property
    def is_sync(self) -> bool:
        return False


class ValueDummy(SyncTemplate):
    def __init__(self, value, size: int, name: str, **kwargs):
        super().__init__(name=name, manager=None, ver_shm=None)

        self._size = size
        self._value = value

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.removesuffix('Dummy'),
            name=self._name,
            size=self._size,
            data=self._value
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        data = json_dict['data']

        self = cls(
            name=json_dict['name'],
            size=json_dict['size'],
            value=data
        )

        return self

    def to_shm(self, override: bool = False):
        pass

    def from_shm(self, override: bool = False):
        pass

    def unlink(self):
        super().unlink()

        self.clear()

    def clear(self, silent: bool = False):
        pass

    @property
    def is_sync(self) -> bool:
        return False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


IntValueDummy = ValueDummy
FloatValueDummy = ValueDummy

if os.name == 'posix':
    from .shm_manager_posix import *
elif os.name == 'nt':
    LOGGER.warning(f'Shared memory support is limited on {platform.system()}-{platform.release()}-{platform.machine()}.')
    from .shm_manager_nt import *

__all__ = ['SharedMemoryCore', 'CachedMemoryCore', 'SyncMemoryCore',
           'Vector', 'NamedVector', 'Deque', 'IntValue', 'FloatValue',
           'VectorDummy', 'NamedVectorDummy', 'DequeDummy', 'ValueDummy']
