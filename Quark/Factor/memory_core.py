from __future__ import annotations

__package__ = 'Quark.Factor'

import abc
import enum
import json
import pickle
import uuid
from collections import deque
from multiprocessing import shared_memory, Lock
from typing import Iterable, Self

import numpy as np

from .. import LOGGER

LOGGER = LOGGER.getChild('memory_core')


class SharedMemoryCore(object):
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def serialize_str_vector(self, vector: list[str]) -> bytes:
        vector_bytes = b'\x00'.join([_.encode(self.encoding) for _ in vector])
        return vector_bytes

    def deserialize_str_vector(self, vector_bytes: bytes) -> list[str]:
        vector = [_.decode(self.encoding) for _ in vector_bytes.split(b'\x00')]
        return vector

    def init_buffer(self, name: str, buffer_size: int) -> shared_memory.SharedMemory:
        try:
            shm = shared_memory.SharedMemory(name=name)

            if shm.size != buffer_size:
                shm.close()
                shm.unlink()

                shm = shared_memory.SharedMemory(name=name, create=True, size=buffer_size)
        except FileNotFoundError as _:
            shm = shared_memory.SharedMemory(name=name, create=True, size=buffer_size)

        return shm

    def get_buffer(self, name: str) -> shared_memory.SharedMemory | None:
        try:
            shm = shared_memory.SharedMemory(name=name)
        except FileNotFoundError as _:
            shm = None

        return shm

    def set_int(self, name: str, value: int, buffer_size: int = 8) -> shared_memory.SharedMemory:
        shm = self.init_buffer(name=name, buffer_size=buffer_size)
        shm.buf[:] = value.to_bytes(length=buffer_size, signed=True)
        return shm

    def get_int(self, name: str, default: int = None) -> int | None:
        shm = self.get_buffer(name=name)
        value = default if shm is None else int.from_bytes(shm.buf, signed=True)
        return value

    def set_vector(self, name: str, vector: list[float | int | bool] | np.ndarray) -> shared_memory.SharedMemory:
        arr = np.array(vector)
        shm = self.init_buffer(name=name, buffer_size=arr.nbytes)
        shm.buf[:] = bytes(arr.data)
        return shm

    def get_vector(self, name: str, target: list = None) -> list[float]:
        shm = self.get_buffer(name=name)
        vector = [] if shm is None else np.ndarray(shape=(-1,), buffer=shm.buf).tolist()

        if target is None:
            return vector
        else:
            target.clear()
            target.extend(vector)
            return target

    def set_str_vector(self, name: str, vector: list[str]) -> shared_memory.SharedMemory:
        vector_bytes = self.serialize_str_vector(vector)
        shm = self.init_buffer(name=name, buffer_size=len(vector_bytes))
        shm.buf[:] = vector_bytes
        return shm

    def get_str_vector(self, name: str, target: list = None) -> list[float]:
        shm = self.get_buffer(name=name)
        vector = [] if shm is None else self.deserialize_str_vector(vector_bytes=bytes(shm.buf))

        if target is None:
            return vector
        else:
            target.clear()
            target.extend(vector)
            return target

    def set_named_vector(self, name: str, obj: dict[str, float]) -> None:
        """
        sync the dict[str, float]
        """
        keys_name = f'{name}.keys'
        values_name = f'{name}.values'

        self.set_str_vector(name=keys_name, vector=list(obj.keys()))
        self.set_vector(name=values_name, vector=list(obj.values()))

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

    def sync(self, name: str, obj: ...) -> shared_memory.SharedMemory:
        serialized = pickle.dumps(obj)
        shm = self.init_buffer(name=name, buffer_size=len(serialized))
        shm.buf[:] = serialized

        return shm

    def get(self, name: str, default=None) -> ...:
        shm = self.get_buffer(name=name)
        obj = default if shm is None else pickle.loads(bytes(shm.buf))

        return obj


class CachedMemoryCore(SharedMemoryCore):
    def __init__(self, prefix: str, encoding='utf-8'):
        super().__init__(encoding=encoding)

        self.prefix = prefix

        self.shm_size: dict[str, shared_memory.SharedMemory] = {}
        self.shm_cache: dict[str, shared_memory.SharedMemory] = {}

    @classmethod
    def cache_name(cls, name: str) -> str:
        return f'{name}._cache_size'

    def get_buffer(self, name: str = None, real_name: str = None) -> shared_memory.SharedMemory | None:

        if real_name is None and name is None:
            raise ValueError('Must assign a "name" or "real_name",')
        elif real_name is None:
            real_name = f'{self.prefix}.{name}'

        # get the shm storing the size data
        if real_name in self.shm_size:
            shm_size = self.shm_size[real_name]
        else:
            shm_size = super().get_buffer(name=self.cache_name(real_name))

            # no size info found
            # since the get_buffer should be called after init_buffer, thus the shm_size should be initialized before this function called.
            # this should not happen. an error message is generated.
            # no cache info stored
            if shm_size is None:
                LOGGER.error(f'Shared memory "{self.cache_name(real_name)}" not found! Expect a 8 bytes shared memory.')
                return super().get_buffer(name=real_name)

            self.shm_size[real_name] = shm_size

        cache_size = int.from_bytes(shm_size.buf)

        # cache not hit. This could happen in the child-processes.
        if name not in self.shm_cache:
            shm = super().get_buffer(name=real_name)

            # for the similar reason above, the get_buffer should be called after init_buffer
            # shm should never be None
            if shm is None:
                LOGGER.error(f'Shared memory "{real_name}" not found!, you should call init_buffer first')
            else:
                self.shm_cache[real_name] = shm

            return shm
        shm = self.shm_cache[real_name]

        # the cache size is the requested size, cache validated.
        if shm.size == cache_size:
            return shm

        # the cache size does not match the requested size, cache validation failed, this could be a result of lack of lock.
        shm = super().get_buffer(name=real_name)
        # the get-process should not update the size log, this is the tasks for the process altering shared memory.
        # shm_size.buf[:] = shm.size.to_bytes(length=8)
        return shm

    def init_buffer(self, buffer_size: int, name: str = None, real_name: str = None) -> shared_memory.SharedMemory:
        if real_name is None and name is None:
            raise ValueError('Must assign a "name" or "real_name",')
        elif real_name is None:
            real_name = f'{self.prefix}.{name}'

        # cache size log found in local
        if real_name in self.shm_size:
            shm_size = self.shm_size[real_name]
            cache_size = int.from_bytes(shm_size.buf)

            # since the cache size info exist, the shm must exist, in ether memory or local or both.
            if real_name in self.shm_cache:
                shm = self.shm_cache[real_name]
            else:
                shm = self.shm_cache[real_name] = super().get_buffer(name=real_name)

            if shm is None:
                shm = self.shm_cache[real_name] = super().init_buffer(name=real_name, buffer_size=buffer_size)
                LOGGER.warning(f'{real_name} cache size info found, but the shared memory does not exist. This might caused by force killing child processes.')

            # cache hit
            if shm.size == buffer_size == cache_size:
                return shm

            # cache not hit, unlink the original shm and create a new one
            shm.close()
            shm.unlink()
            shm_size.buf[:] = buffer_size.to_bytes(length=8)
            shm = self.shm_cache[real_name] = super().init_buffer(name=real_name, buffer_size=buffer_size)
            return shm
        # cache size info not found
        elif (shm_size := super().get_buffer(name=self.cache_name(real_name))) is None:
            self.shm_size[real_name] = shm_size = super().init_buffer(name=self.cache_name(real_name), buffer_size=8)
            shm_size.buf[:] = buffer_size.to_bytes(length=8)

            # no cache size info but still have a cached obj, this should never happen
            if real_name in self.shm_cache:
                raise ValueError('Cache found but no cache size info found, potential collision on shared memory names, stop and exit is advised.')

            shm = self.shm_cache[real_name] = super().init_buffer(name=real_name, buffer_size=buffer_size)
            return shm
        # cache size found in memory: update local logs and re-run
        else:
            self.shm_size[real_name] = shm_size
            # avoid issues in nested-inheritance
            return CachedMemoryCore.init_buffer(self=self, real_name=real_name, buffer_size=buffer_size)


class SyncMemoryCore(CachedMemoryCore):
    class SyncTypes(enum.Enum):
        Vector = 'Vector'
        NamedVector = 'NamedVector'
        Deque = 'Deque'
        Value = 'Value'

    def __init__(self, prefix: str = None, encoding='utf-8', dummy: bool = False):
        if prefix is None:
            prefix = uuid.uuid4().hex

        super().__init__(prefix=prefix, encoding=encoding)

        self.dummy: bool = dummy

        self.lock = Lock() if not self.dummy else None
        self.storage: dict[str, SyncTemplate] = {}

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def get_buffer(self, name: str = None, real_name: str = None) -> shared_memory.SharedMemory | None:
        with self.lock:
            return super().get_buffer(name=name, real_name=real_name)

    def init_buffer(self, buffer_size: int, name: str = None, real_name: str = None, ) -> shared_memory.SharedMemory:
        with self.lock:
            return super().init_buffer(name=name, real_name=real_name, buffer_size=buffer_size)

    def register(self, *args, name: str, dtype: str | SyncTypes, use_cache: bool = True, **kwargs):
        if isinstance(dtype, self.SyncTypes):
            dtype = dtype.value

        if use_cache and name in self.storage:
            sync_storage = self.storage[name]
            if dtype == sync_storage.__class__.__name__:
                return self.storage[name]
            else:
                LOGGER.warning(f'Name {name} already registered as {sync_storage.__class__.__name__}. Override with new sync memory type {dtype} may cause conflicts. Use with caution.')
                LOGGER.warning('This action will unlink the original one.')
                sync_storage.unlink()

        if self.dummy:
            sync_storage = self.register_dummy(name=name, dtype=dtype, *args, **kwargs)
        else:
            match dtype:
                case 'NamedVector':
                    sync_storage = NamedVector(manager=self, name=name, *args, **kwargs)
                case 'Vector':
                    sync_storage = Vector(manager=self, name=name, *args, **kwargs)
                case 'Deque':
                    sync_storage = Deque(manager=self, name=name, *args, **kwargs)
                case 'IntValue':
                    sync_storage = IntValue(manager=self, name=name, *args, **kwargs)
                case 'FloatValue':
                    sync_storage = FloatValue(manager=self, name=name, *args, **kwargs)
                case _:
                    raise ValueError(f'Invalid dtype {dtype}.')

        self.storage[name] = sync_storage
        return sync_storage

    def register_dummy(self, *args, name: str, dtype: str | SyncTypes, **kwargs):
        if not self.dummy:
            LOGGER.warning('Using register_dummy may override the synchronized storage. This function is for testing only.')

        match dtype:
            case 'NamedVector':
                sync_storage = NamedVectorDummy(manager=self, name=name, *args, **kwargs)
            case 'Vector':
                sync_storage = VectorDummy(manager=self, name=name, *args, **kwargs)
            case 'Deque':
                sync_storage = DequeDummy(manager=self, name=name, *args, **kwargs)
            case 'IntValue':
                sync_storage = ValueDummy(manager=self, name=name, *args, **kwargs)
            case _:
                raise ValueError(f'Invalid dtype {dtype}.')

        self.storage[name] = sync_storage
        return sync_storage

    def unlink(self):

        for name, sync_storage in self.storage.items():
            LOGGER.info(f'Unlinking {sync_storage.__class__.__name__} {name} buffer...')

            try:
                sync_storage.unlink()
            except FileNotFoundError as _:
                continue

        LOGGER.info(f'{len(self.storage)} storage entries clean up complete!')
        self.storage.clear()

        for buffer in self.shm_size.values():
            try:
                buffer.close()
                buffer.unlink()
            except FileNotFoundError as _:
                continue

        for buffer in self.shm_cache.values():
            try:
                buffer.close()
                buffer.unlink()
            except FileNotFoundError as _:
                continue

        LOGGER.info(f'{len(self.shm_size)} cache entries clean up complete!')
        self.shm_size.clear()
        self.shm_cache.clear()

    def to_shm(self, override: bool = False):
        for sync_storage in self.storage.values():
            is_sync = sync_storage.is_sync and not override

            if not is_sync:
                sync_storage.to_shm()

    def from_shm(self, override: bool = False):
        for sync_storage in self.storage.values():
            is_sync = sync_storage.is_sync and not override

            if not is_sync:
                sync_storage.from_shm()

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            encoding=self.encoding,
            prefix=self.prefix,
            storage={name: storage.to_json(fmt='dict') for name, storage in self.storage.items()},
            dummy=self.dummy
        )

        # self.to_shm(override=True)

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            encoding=json_dict['encoding'],
            prefix=json_dict['prefix'],
            dummy=json_dict['dummy']
        )

        self.storage = {name: SyncTemplate.from_json(manager=self, json_message=data) for name, data in json_dict['storage'].items()},

        # self.from_shm(override=True)
        return self


class SyncTemplate(object, metaclass=abc.ABCMeta):
    def __init__(self, manager: SharedMemoryCore, name: str):
        self._name = name
        self._manager = manager
        self._ver_shm = manager.init_buffer(name=f'{name}.ver_code', buffer_size=16)
        self._ver_local: bytes = uuid.uuid4().bytes
        self.desync_counter = 0

    @abc.abstractmethod
    def to_json(self, fmt='str'):
        ...

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict):
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        dtype = json_dict['dtype']

        if manager.dummy:
            match dtype:
                case 'NamedVector':
                    self = NamedVectorDummy.from_json(manager=manager, json_message=json_dict)
                case 'Vector':
                    self = VectorDummy.from_json(manager=manager, json_message=json_dict)
                case 'Deque':
                    self = DequeDummy.from_json(manager=manager, json_message=json_dict)
                case 'IntValue':
                    self = ValueDummy.from_json(manager=manager, json_message=json_dict)
                case _:
                    raise ValueError(f'Invalid dtype {dtype}.')
        else:
            match dtype:
                case 'NamedVector':
                    self = NamedVector.from_json(manager=manager, json_message=json_dict)
                case 'Vector':
                    self = Vector.from_json(manager=manager, json_message=json_dict)
                case 'Deque':
                    self = Deque.from_json(manager=manager, json_message=json_dict)
                case 'IntValue':
                    self = IntValue.from_json(manager=manager, json_message=json_dict)
                case 'FloatValue':
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

    def unlink(self):
        try:
            self._ver_shm.close()
            self._ver_shm.unlink()
        except FileNotFoundError as _:
            pass

    def new_ver(self, desync_warning_limit: int = 100):
        self._ver_local: bytes = uuid.uuid4().bytes

        self.desync_counter += 1

        if desync_warning_limit is not None and self.desync_counter >= desync_warning_limit > 0:
            LOGGER.error(f'{self} de-synchronize too many times, check the code!')

    def set_ver(self):
        self._ver_shm.buf[:] = self._ver_local

    def get_ver(self):
        self._ver_local = bytes(self._ver_shm.buf)

    @abc.abstractmethod
    def clear(self, silent: bool = False):
        ...

    @property
    def name(self):
        return self._name

    @property
    def encoding(self):
        return self._manager.encoding

    @property
    def ver_shm(self) -> bytes:
        """
        This is the version code of data in shared memory object.

        for each breaking update, the version code will be added 1.
        """
        if (buffer := self._ver_shm.buf) is None:
            LOGGER.warning(f'ver_code of {self.name} not found in shared memory, this might be caused by called the unlink() method.')
            return b''
        return bytes(buffer)

    @property
    def ver_local(self) -> bytes:
        return self._ver_local

    @property
    def is_sync(self) -> bool:
        """
        A bool indicating if the data is connected to the shared memory
        Returns:

        """
        if self._ver_shm.buf is None:
            return False
        elif self.ver_local == self.ver_shm:
            return True
        return False


class NamedVector(SyncTemplate):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, **kwargs):
        data = dict(*args, **kwargs)

        super().__init__(name=name, manager=manager)
        self._dtype = np.double
        self._keys: list[str] = list(data.keys())
        self._idx: dict[str, int] = {key: idx for idx, key in enumerate(self._keys)}
        self._values: np.ndarray = np.array(list(data.values()), dtype=self._dtype)

        self._shm_keys: shared_memory.SharedMemory | None = None
        self._shm_values: shared_memory.SharedMemory | None = None

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            name=self._name,
            data=dict(self.items())
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            manager=manager,
            name=json_dict['name']
        )

        return self

    def to_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if not is_sync or self._shm_keys is None or self._shm_values is None:
            self._shm_keys = self._manager.set_str_vector(name=self.name_keys, vector=self._keys)
            self._shm_values = self._manager.set_vector(name=self.name_values, vector=self._values)
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_keys is None or self._shm_values is None:
            self._shm_keys = self._manager.get_buffer(name=self.name_keys)
            self._shm_values = self._manager.get_buffer(name=self.name_values)
            self._keys = self._manager.deserialize_str_vector(vector_bytes=bytes(self._shm_keys.buf))
            self._idx = {key: idx for idx, key in enumerate(self._keys)}
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self.get_ver()
            return

    def __setitem__(self, key: str, value: float):
        if key not in self._idx:
            self._idx[key] = len(self._keys)
            self._keys.append(key)
            self._values = np.append(self._values, value)
            self.new_ver()
        else:
            idx = self._idx[key]
            self._values[idx] = value

    def __getitem__(self, key: str):
        if key not in self._idx:
            raise KeyError(f'Key {key} not found')

        return self._values[self._idx[key]]

    def __contains__(self, key: str):
        return self._idx.__contains__(key)

    def __len__(self):
        return len(self._idx)

    def __bool__(self):
        return bool(self._idx)

    def __iter__(self):
        return self._idx.__iter__()

    def get(self, key: str, default_value: float = None):
        if key not in self._idx:
            return default_value
        else:
            return self._values[self._idx[key]]

    def update(self, data_dict: dict[str, float] = None, **kwargs):
        if data_dict is not None:
            for key, value in data_dict.items():
                self.__setitem__(key=key, value=value)

        for key, value in kwargs.items():
            self.__setitem__(key=key, value=value)

    def pop(self, key: str, default_value: float = None):
        if key not in self._idx:
            return default_value
        else:
            idx = self._idx.pop(key)
            value = self._values[idx]
            self._values = np.delete(self._values, idx)
            self.new_ver()
            return value

    def keys(self) -> list[str]:
        return list(self._keys)

    def values(self) -> list[float]:
        return self._values.tolist()

    def items(self):
        return zip(self._keys, self._values)

    def clear(self, silent: bool = False):
        if not silent:
            LOGGER.warning(f'{self.__class__.__name__} should not be clear. If it is to free the resource, use unlink().')

        self._keys.clear()
        self._idx.clear()
        self._values = np.array([], dtype=self._dtype)
        self.new_ver()

    def unlink(self):
        super().unlink()

        if self._shm_keys is not None:
            try:
                self._shm_keys.close()
                self._shm_keys.unlink()
            except FileNotFoundError as _:
                pass

        if self._shm_values is not None:
            try:
                self._shm_values.close()
                self._shm_values.unlink()
            except FileNotFoundError as _:
                pass

        self.clear(silent=True)

    @property
    def name_keys(self) -> str:
        return f'{self._name}.keys'

    @property
    def name_values(self) -> str:
        return f'{self._name}.values'


class Vector(SyncTemplate):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, **kwargs):
        data = list(*args, **kwargs)

        super().__init__(name=name, manager=manager)
        self._dtype = np.double
        self._values = np.array(data, dtype=self._dtype)

        self._shm_values: shared_memory.SharedMemory | None = None

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            name=self._name,
            data=self._values.tolist()
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            manager=manager,
            name=json_dict['name']
        )

        return self

    def to_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if not is_sync or self._shm_values is None:
            self._shm_values = self._manager.set_vector(name=self.name, vector=self._values)
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_values is None:
            self._shm_values = self._manager.get_buffer(name=self.name)
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self.get_ver()
            return

    def __setitem__(self, index: int, value: float):
        self._values.__setitem__(index, value)

    def __getitem__(self, index: int):
        return self._values.__getitem__(index)

    def __contains__(self, value: float):
        return self._values.__contains__(value)

    def __len__(self):
        return self._values.__len__()

    def __bool__(self):
        return self._values.__bool__()

    def __iter__(self):
        return self._values.__iter__()

    def append(self, value: float):
        self._values = np.append(self._values, value)
        self.new_ver()

    def pop(self, idx) -> float:
        value = self._values[idx]
        self._values = np.delete(self._values, idx)
        self.new_ver()
        return value

    def extend(self, data: Iterable[float]):
        # noinspection PyTypeChecker
        self._values = np.append(self._values, data)
        self.new_ver()

    def clear(self, silent: bool = False):
        if not silent:
            LOGGER.warning(f'{self.__class__.__name__} should not be clear. If it is to free the resource, use unlink().')

        self._values = np.array([], dtype=self._dtype)
        self.new_ver()

    def unlink(self):
        super().unlink()

        if self._shm_values is not None:
            try:
                self._shm_values.close()
                self._shm_values.unlink()
            except FileNotFoundError as _:
                pass

        self.clear(silent=True)


class Deque(SyncTemplate):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, maxlen: int, **kwargs):
        data = list(*args, **kwargs)

        super().__init__(name=name, manager=manager)
        self._maxlen = maxlen
        self._dtype = np.double
        self._values = np.zeros(shape=maxlen, dtype=self._dtype)
        self._length = min(maxlen, len(data))

        if self._length:
            self._values[-self._length:] = data[-self._length:]

        self._shm_values: shared_memory.SharedMemory | None = None
        self._shm_length: shared_memory.SharedMemory | None = None

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            name=self._name,
            maxlen=self._maxlen,
            data=self.data.tolist()
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            manager=manager,
            name=json_dict['name'],
            maxlen=json_dict['maxlen']
        )

        return self

    def to_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if not is_sync or self._shm_values is None or self._shm_length is None:
            self._shm_values = self._manager.set_vector(name=self.name, vector=self._values)
            self._shm_length = self._manager.set_int(name=self.name_length, value=self._length)
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_values is None or self._shm_length is None:
            self._shm_values = self._manager.get_buffer(name=self.name)
            self._shm_length = self._manager.get_buffer(name=self.name_length)
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buf)
            self._length = int.from_bytes(self._shm_length.buf)
            self._maxlen = len(self._values)
            self.get_ver()
            return

    def __setitem__(self, index: int, value: float):
        self.data.__setitem__(index, value)

    def __getitem__(self, index: int):
        return self.data.__getitem__(index)

    def __contains__(self, value: float):
        return self.data.__contains__(value)

    def __len__(self):
        return self.length

    def __bool__(self):
        return bool(self.__len__())

    def __iter__(self):
        return self.data.__iter__()

    def append(self, value: float):
        if self.length < self._maxlen:
            self.length += 1

        self._values[:-1] = self._values[1:]
        self._values[-1] = value

    def pop(self) -> float:
        if not self.length:
            raise IndexError('pop from an empty deque')

        # noinspection PyTypeChecker
        value: float = self._values[-1]
        self.length -= 1
        self._values[1:] = self._values[:-1]
        self._values[0] = 0
        return value

    def extend(self, data: Iterable[float]):
        for value in data:
            self.append(value)

    def clear(self, silent: bool = False):
        if not silent:
            LOGGER.warning(f'{self.__class__.__name__} should not be clear. If it is to free the resource, use unlink().')

        if self.is_sync:
            self._values[:] = np.zeros(shape=self._maxlen, dtype=self._dtype)
            self.length = 0
        else:
            self._values = np.zeros(shape=self._maxlen, dtype=self._dtype)
            self.new_ver()
            self._length = 0

    def unlink(self):
        super().unlink()

        if self._shm_values is not None:
            try:
                self._shm_values.close()
                self._shm_values.unlink()
            except FileNotFoundError as _:
                pass

        if self._shm_length is not None:
            try:
                self._shm_length.close()
                self._shm_length.unlink()
            except FileNotFoundError as _:
                pass

        self.clear(silent=True)

    @property
    def name_length(self) -> str:
        return f'{self.name}.length'

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def data(self) -> np.ndarray:
        return self._values[self.maxlen - self.__len__():]

    @property
    def length(self) -> int:
        if self._shm_length is None:
            return self._length
        else:
            return int.from_bytes(self._shm_length.buf)

    @length.setter
    def length(self, value: int):
        self._length = value

        if self._shm_length:
            self._shm_length.buf[:] = self._length.to_bytes(length=self._shm_length.size)


class IntValue(SyncTemplate):
    def __init__(self, value: int, manager: SharedMemoryCore, name: str, size: int = 8):
        super().__init__(name=name, manager=manager)
        self._size = size
        self._value: int = value

        self._shm: shared_memory.SharedMemory | None = None

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            name=self._name,
            size=self._size,
            data=self.value
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        data = json_dict['data']

        self = cls(
            manager=manager,
            name=json_dict['name'],
            size=json_dict['size'],
            value=data
        )

        return self

    def to_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if not is_sync or self._shm is None:
            self._shm = self._manager.set_int(name=self.name, value=self._value, buffer_size=self._size)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm is None:
            self._shm = self._manager.get_buffer(name=self.name)
            self._value = int.from_bytes(self._shm.buf, signed=True)
            self.get_ver()
            return

    def unlink(self):
        super().unlink()

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except FileNotFoundError as _:
                pass

    def clear(self, silent: bool = False):
        pass

    @property
    def value(self):
        if self._shm is not None:
            self._value = int.from_bytes(self._shm.buf, signed=True)

        return self._value

    @value.setter
    def value(self, new_value: int):
        self._value = new_value
        self.to_shm(override=True)


class FloatValue(SyncTemplate):
    def __init__(self, value: float, manager: SharedMemoryCore, name: str, size: int = 8):
        super().__init__(name=name, manager=manager)

        if size != 8:
            LOGGER.warning(f'numpy double byte size should be 8, not {size} at most of the platforms. Use with caution.')

        self._size = size
        self._value = np.array([value], dtype=np.double)

        self._shm: shared_memory.SharedMemory | None = None

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__,
            name=self._name,
            size=self._size,
            data=self.value
        )

        if fmt == 'dict':
            return data_dict
        elif fmt == 'str':
            return json.dumps(data_dict, **kwargs)
        else:
            raise ValueError(f'Invalid format {fmt}, except "dict" or "str".')

    @classmethod
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        data = json_dict['data']

        self = cls(
            manager=manager,
            name=json_dict['name'],
            size=json_dict['size'],
            value=data
        )

        return self

    def to_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if not is_sync or self._shm is None:
            self._shm = self._manager.init_buffer(name=self.name, buffer_size=self._size)
            self._shm.buf[:] = self._value.tobytes()
            self._value = np.ndarray(shape=(1,), dtype=np.double, buffer=self._shm.buf)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm is None:
            self._shm = self._manager.get_buffer(name=self.name)
            self._value = np.ndarray(shape=(1,), dtype=np.double, buffer=self._shm.buf)
            self.get_ver()
            return

    def unlink(self):
        super().unlink()

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except FileNotFoundError as _:
                pass

    def clear(self, silent: bool = False):
        pass

    @property
    def value(self):
        return self._value[0]

    @value.setter
    def value(self, new_value: int):
        self._value[0] = new_value


class NamedVectorDummy(SyncTemplate, dict):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, **kwargs):
        super().__init__(name=name, manager=manager)
        dict.__init__(self=self, *args, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.rstrip('Dummy'),
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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            manager=manager,
            name=json_dict['name'],
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
        dict.clear(self)

    @property
    def is_sync(self) -> bool:
        return False

    @is_sync.setter
    def is_sync(self, is_sync: bool):
        raise AttributeError(f'Can not set property is_sync in {self.__class__.__name__}.')


class VectorDummy(SyncTemplate, list):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, **kwargs):
        super().__init__(name=name, manager=manager)
        list.__init__(self=self, *args, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.rstrip('Dummy'),
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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            manager=manager,
            name=json_dict['name'],
        )

        self.extend(json_dict['data'])

        return self

    def to_shm(self, override: bool = False):
        pass

    def from_shm(self, override: bool = False):
        pass

    def unlink(self):
        super().unlink()

        self.clear()

    def clear(self, silent: bool = False):
        list.clear(self)

    @property
    def is_sync(self) -> bool:
        return False


class DequeDummy(SyncTemplate, deque):
    def __init__(self, *args, manager: SharedMemoryCore, name: str, **kwargs):
        super().__init__(name=name, manager=manager)
        deque.__init__(self=self, *args, **kwargs)

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.rstrip('Dummy'),
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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        self = cls(
            json_dict['data'],
            manager=manager,
            name=json_dict['name'],
            maxlen=json_dict['maxlen']
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
        deque.clear(self)

    @property
    def is_sync(self) -> bool:
        return False


class ValueDummy(SyncTemplate):
    def __init__(self, value, size: int, manager: SharedMemoryCore, name: str):
        super().__init__(name=name, manager=manager)

        self._size = size
        self._value = value

    def to_json(self, fmt='str', **kwargs) -> str | dict:
        data_dict = dict(
            dtype=self.__class__.__name__.rstrip('Dummy'),
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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict) -> Self:
        if isinstance(json_message, dict):
            json_dict = json_message
        else:
            json_dict = json.loads(json_message)

        data = json_dict['data']

        self = cls(
            manager=manager,
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


__all__ = ['SyncMemoryCore',
           'Vector', 'NamedVector', 'Deque', 'IntValue', 'FloatValue',
           'VectorDummy', 'NamedVectorDummy', 'DequeDummy', 'ValueDummy']


def unittest_named_vector(memory_core):
    LOGGER.info('Test registration for NamedVector:')
    d1 = memory_core.register(name='test_dict', dtype='NamedVector')
    d1['a'] = 1
    d1['b'] = 1.56

    assert d1['a'] == 1
    assert d1['b'] == 1.56
    LOGGER.info(f'd1: {dict(d1)}')
    LOGGER.info('success')

    LOGGER.info('Test serialization:')
    LOGGER.info(f'to_dict: {dict(d1)}')  # this behavior is unexpected, it seems that the dict constructor trys the .items() method first.
    LOGGER.info(f'to_json: {d1.to_json()}')
    d2: NamedVector = SyncTemplate.from_json(manager=memory_core, json_message=d1.to_json())
    LOGGER.info(f'from_json: {dict(d2)}')
    assert d1.keys() == d2.keys() and d1.values() == d2.values()
    LOGGER.info('success')

    LOGGER.info('Test synchronization:')
    assert (not d1.is_sync) and (not d2.is_sync)
    d1.to_shm()
    assert d1.is_sync
    d3 = memory_core.register({'c': -29.5}, name='test_dict', dtype='NamedVector', use_cache=False)
    assert not d3.is_sync
    d3.from_shm()
    assert d3.is_sync and d1.is_sync
    d3['d'] = -np.inf
    assert (not d3.is_sync) and d1.is_sync
    d3.to_shm()
    assert (not d1.is_sync) and d3.is_sync
    d1.from_shm()
    d2.from_shm()
    d2['b'] = 1843.4
    d1['a'] = np.nan
    d3['d'] = 0
    assert d1.is_sync and d2.is_sync and d3.is_sync
    LOGGER.info(f'all dict should be the same:\n{dict(d1)}\n{dict(d2)}\n{dict(d3)}')
    LOGGER.info('success')
    d1.unlink()
    d2.unlink()
    d3.unlink()
    LOGGER.info('unlink successful')


def unittest_vector(memory_core):
    LOGGER.info('Test registration for Vector:')
    v1 = memory_core.register(name='test_vector', dtype='Vector')
    v1.append(1.)
    v1.extend([3, np.inf, -5, 10.987])
    LOGGER.info(f'v1: {list(v1)}')
    LOGGER.info('success')

    LOGGER.info('Test serialization:')
    LOGGER.info(f'to_json: {v1.to_json()}')
    v2: Vector = SyncTemplate.from_json(manager=memory_core, json_message=v1.to_json())
    LOGGER.info(f'from_json: {list(v2)}')
    assert list(v1) == list(v2)
    LOGGER.info('success')

    LOGGER.info('Test synchronization:')
    assert (not v1.is_sync) and (not v2.is_sync)
    v1.to_shm()
    assert v1.is_sync and (not v2.is_sync)
    v3 = memory_core.register([-1, 0, np.nan], name='test_vector', dtype='Vector', use_cache=False)
    assert not v3.is_sync
    v3.from_shm()
    assert v1.is_sync and (not v2.is_sync) and v3.is_sync
    assert list(v1) == list(v2) == list(v3)
    v3[-1] = -np.inf
    assert v1.is_sync and (not v2.is_sync) and v3.is_sync
    v2.to_shm()
    assert (not v1.is_sync) and v2.is_sync and (not v3.is_sync)
    v1.from_shm()
    v3.from_shm()
    assert v1.is_sync and v2.is_sync and v3.is_sync
    v1[0] = 10
    v2[1] = np.inf
    v3[3] = -100.1
    assert v1.is_sync and v2.is_sync and v3.is_sync
    assert list(v1) == list(v2) == list(v3)
    LOGGER.info(f'all vector should be the same:\n{list(v1)}\n{list(v2)}\n{list(v3)}')
    LOGGER.info('success')
    v1.unlink()
    v2.unlink()
    v3.unlink()
    LOGGER.info('unlink successful')


def unittest_deque(memory_core):
    LOGGER.info('Test registration for Deque:')
    dq1: Deque = memory_core.register(name='test_deque', dtype='Deque', maxlen=5)
    dq1.append(1.)
    assert len(dq1) == 1 and dq1[-1] == 1.
    dq1.extend([3, np.inf, -5, 10.987, 908])
    assert len(dq1) == 5
    assert dq1.data.tolist() == [3, np.inf, -5, 10.987, 908]
    LOGGER.info(f'dq1: {list(dq1)}')
    LOGGER.info('success')

    LOGGER.info('Test serialization:')
    LOGGER.info(f'to_json: {dq1.to_json()}')
    dq2: Vector = SyncTemplate.from_json(manager=memory_core, json_message=dq1.to_json())
    LOGGER.info(f'from_json: {list(dq2)}')
    assert list(dq1) == list(dq2)
    LOGGER.info('success')

    LOGGER.info('Test synchronization:')
    dq1.clear()
    dq2.clear()
    assert (not dq1.is_sync) and (not dq2.is_sync)
    dq1.append(1.5)
    dq1.to_shm()
    assert dq1.is_sync and (not dq2.is_sync)
    dq2.from_shm()
    assert dq1.is_sync and dq2.is_sync
    assert len(dq2) == 1 and dq2[-1] == 1.5
    dq2[-1] = 3
    assert len(dq1) == 1 and dq1[-1] == 3
    dq2.append(-10)
    assert len(dq1) == 2 and dq1[-1] == -10
    dq1.append(np.inf)
    assert len(dq2) == 3 and dq2[-1] == np.inf
    dq3: Deque = memory_core.register([-1, 0, np.nan], name='test_deque', dtype='Deque', maxlen=10, use_cache=False)
    assert not dq3.is_sync
    dq3.from_shm()
    assert dq3.maxlen == 5 and list(dq1) == list(dq3)
    dq3[-1] = -np.inf
    assert dq1.is_sync and dq2.is_sync and dq3.is_sync
    dq1[0] = 10
    dq2[1] = np.inf
    dq3[2] = -100.1
    dq1.extend([1, -2])
    dq2.append(10)
    assert dq3.pop() == 10
    assert dq3[-1] == -2
    assert dq1.is_sync and dq2.is_sync and dq3.is_sync
    assert list(dq1) == list(dq2) == list(dq3)
    LOGGER.info(f'all deque should be the same:\n{list(dq1)}\n{list(dq2)}\n{list(dq3)}')
    LOGGER.info('success')

    dq1.unlink()
    dq2.unlink()
    dq3.unlink()
    LOGGER.info('unlink successful')


def unittest_int_value(memory_core):
    LOGGER.info('Test registration for IntValue:')
    iv1: IntValue = memory_core.register(name='test_int_value', dtype='IntValue', value=10)
    assert iv1.value == 10 and not iv1.is_sync
    iv1.value = -20
    assert iv1.value == -20 and iv1.is_sync
    LOGGER.info(f'iv1: {iv1.value}')
    LOGGER.info('success')

    LOGGER.info('Test serialization:')
    LOGGER.info(f'to_json: {iv1.to_json()}')
    iv2: IntValue = SyncTemplate.from_json(manager=memory_core, json_message=iv1.to_json())
    LOGGER.info(f'from_json: {iv2.value}')
    assert iv1.value == iv2.value
    LOGGER.info('success')

    LOGGER.info('Test synchronization:')
    assert iv1.is_sync and not iv2.is_sync
    iv2.from_shm()
    assert iv1.is_sync and iv2.is_sync
    iv2.value = -135
    assert iv1.value == iv2.value == -135 and iv1.is_sync and iv2.is_sync
    iv1.value = 100
    assert iv1.value == iv2.value == 100 and iv1.is_sync and iv2.is_sync
    LOGGER.info('success')

    iv1.unlink()
    iv2.unlink()
    LOGGER.info('unlink successful')


def unittest_float_value(memory_core):
    LOGGER.info('Test registration for FloatValue:')
    iv1: FloatValue = memory_core.register(name='test_float_value', dtype='FloatValue', value=10.4)
    assert iv1.value == 10.4 and not iv1.is_sync
    iv1.value = -20.5
    assert iv1.value == -20.5 and not iv1.is_sync
    LOGGER.info(f'iv1: {iv1.value}')
    LOGGER.info('success')

    LOGGER.info('Test serialization:')
    LOGGER.info(f'to_json: {iv1.to_json()}')
    iv2: FloatValue = SyncTemplate.from_json(manager=memory_core, json_message=iv1.to_json())
    LOGGER.info(f'from_json: {iv2.value}')
    assert iv1.value == iv2.value
    LOGGER.info('success')

    LOGGER.info('Test synchronization:')
    iv1.to_shm()
    assert iv1.is_sync and not iv2.is_sync
    iv2.from_shm()
    assert iv1.is_sync and iv2.is_sync
    iv2.value = -np.inf
    assert iv1.value == iv2.value == -np.inf and iv1.is_sync and iv2.is_sync
    iv1.value = 10.9187
    assert iv1.value == iv2.value == 10.9187 and iv1.is_sync and iv2.is_sync
    LOGGER.info('success')

    iv1.unlink()
    iv2.unlink()
    LOGGER.info('unlink successful')


def main():
    import os

    memory_core = SyncMemoryCore(prefix='', dummy=False)

    # --- test named vector ---
    unittest_named_vector(memory_core=memory_core)

    # --- test vector ---
    unittest_vector(memory_core=memory_core)

    # --- test deque ---
    unittest_deque(memory_core=memory_core)

    # --- test static values ---
    unittest_int_value(memory_core=memory_core)
    unittest_float_value(memory_core=memory_core)
    memory_core.unlink()
    os._exit(0)


if __name__ == '__main__':
    main()
