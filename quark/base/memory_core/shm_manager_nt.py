import abc
import ctypes
import json
import mmap
import os
import pathlib
import pickle
import uuid
from collections.abc import Iterable
from typing import Self

import numpy as np

from . import LOGGER, MemoryManager, MemoryBuffer as MemoryBufferBase, SyncTypes, SyncTemplate as SyncTemplateBase, NamedVectorDummy, VectorDummy, DequeDummy, ValueDummy


class MemoryBuffer(MemoryBufferBase):
    def __init__(self, name: str, shm_handler: int, buffer: memoryview, size: int):
        super().__init__(name=name, buffer=buffer)
        self.shm_handler = shm_handler
        self._size = size

    def to_bytes(self):
        return bytes(self.buffer[:self._size])

    def to_int(self):
        return int.from_bytes(self.buffer[:self._size])

    def to_float(self):
        return float(np.frombuffer(self.buffer[:self._size])[0])

    def to_str(self, encoding='utf8'):
        return self.buffer[:self._size].tobytes().decode(encoding)

    def to_array(self, shape: tuple[int, ...] = None, dtype: type = None) -> np.ndarray:
        if shape is None:
            shape = (-1,)

        return np.ndarray(shape=shape, dtype=dtype, buffer=self.buffer[:self._size])

    def unpickle(self):
        return pickle.loads(self.buffer[:self._size])

    @property
    def size(self):
        return self._size


class SharedMemoryCore(MemoryManager):
    def __init__(self, shm_lib: str | pathlib.Path = None, **kwargs):
        if os.name != 'nt':
            raise NotImplementedError("Only supported Windows nt platform.")

        super().__init__(**kwargs)
        self.shm_manager = self._init_memory_manager(shm_lib=shm_lib)
        # in windows platform, registering and unlinking the shared memory is extremely difficult.
        # a large enough shm can avoid these issues with ease.
        # recommending passing a large enough value for page_size
        self.page_size = kwargs.get('page_size', mmap.PAGESIZE)

        self.ptr_to_buffer = ctypes.pythonapi.PyMemoryView_FromMemory
        self.ptr_to_buffer.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
        self.ptr_to_buffer.restype = ctypes.py_object

    @classmethod
    def _init_memory_manager(cls, shm_lib: str | pathlib.Path = None):
        if shm_lib is None:
            here = pathlib.Path(__file__).parent.absolute()
            is_found = False
            _file = 'shm.so'

            for _file in os.listdir(here):
                if _file.startswith('shm') and _file.endswith('.dll'):
                    is_found = True
                    break

            if not is_found:
                raise ValueError("Could not find shm library.")

            shm_lib = here.joinpath(_file)

        lib = ctypes.CDLL(str(shm_lib))

        # HANDLE shm_register(const char *key, size_t size)
        lib.shm_register.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.shm_register.restype = ctypes.c_void_p

        # size_t shm_size(HANDLE hMapFile)
        lib.shm_size.argtypes = [ctypes.c_void_p]
        lib.shm_size.restype = ctypes.c_size_t

        # HANDLE shm_get(const char *key)
        lib.shm_get.argtypes = [ctypes.c_char_p]
        lib.shm_get.restype = ctypes.c_void_p

        # void* shm_buffer(HANDLE hMapFile, size_t size)
        lib.shm_buffer.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        lib.shm_buffer.restype = ctypes.c_void_p

        # int shm_close(HANDLE hMapFile)
        lib.shm_close.argtypes = [ctypes.c_void_p]
        lib.shm_close.restype = ctypes.c_int

        # int shm_unregister(const char *key)
        lib.shm_unregister.argtypes = [ctypes.c_char_p]
        lib.shm_unregister.restype = ctypes.c_int

        # int shm_bytes_write(void *shmaddr, const void *data, size_t size)
        lib.shm_bytes_write.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.shm_bytes_write.restype = ctypes.c_int

        # int shm_bytes_read(void *shmaddr, void *buffer, size_t size)
        lib.shm_bytes_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.shm_bytes_read.restype = ctypes.c_int

        # int shm_int_write(void *shmaddr, int value)
        lib.shm_int_write.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.shm_int_write.restype = ctypes.c_int

        # int shm_int_read(void *shmaddr, int *buffer)
        lib.shm_int_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.shm_int_read.restype = ctypes.c_int

        # int shm_dbl_write(void *shmaddr, double value)
        lib.shm_dbl_write.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.shm_dbl_write.restype = ctypes.c_int

        # int shm_dbl_read(void *shmaddr, double *buffer)
        lib.shm_dbl_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        lib.shm_dbl_read.restype = ctypes.c_int

        return lib

    def _get_shm_handler(self, name: str) -> int | None:
        shm_key = name.encode(self.encoding)
        shm_handler = self.shm_manager.shm_get(shm_key)

        if shm_handler is None:
            LOGGER.debug(f'Could not find shm {name}!')

        return shm_handler

    def _get_buffer(self, shm_handler: int, size: int = None) -> memoryview:
        if size is None or size == -1:
            size = self.shm_manager.shm_size(shm_handler)

        if size == -1:
            raise FileNotFoundError(f'Could not determine shm {shm_handler} size!')

        addr = self.shm_manager.shm_buffer(shm_handler, size)
        buffer: memoryview = self.ptr_to_buffer(ctypes.c_void_p(addr), ctypes.c_ssize_t(size), ctypes.c_int(0x200))
        return buffer

    def _set_buffer(self, shm_handler: int, data: bytes, buffer_size: int = None, override: bool = True) -> None:
        if buffer_size is None:
            buffer_size = self.shm_manager.shm_size(shm_handler)

        data_size = len(data)

        if data_size > buffer_size:
            raise ValueError('Data {data_size} exceeds buffer size {buffer_size}!')
        elif data_size < buffer_size and override:
            data = data.ljust(buffer_size, b'\x00')

        res = self.shm_manager.shm_bytes_write(shm_handler, data, len(data))

        if res == -1:
            raise ValueError(f'Shared memory {shm_handler} could not be set with value {data}!')

    def init_buffer(self, name: str, buffer_size: int = None, init_value: bytes = None) -> MemoryBuffer:
        shm_key = name.encode(self.encoding)

        if buffer_size is None:
            buffer_size = len(init_value)

        _alloc = divmod(buffer_size, self.page_size)
        alloc_page = _alloc[0] + (1 if _alloc[1] else [0])
        alloc_size = alloc_page * self.page_size

        shm_handler = self.shm_manager.shm_register(shm_key, alloc_size)

        if shm_handler is None:
            LOGGER.debug(f'Shared memory {name} might be already registered!')
            _shm_handler = self.shm_manager.shm_get(shm_key)

            if _shm_handler is None:
                raise ValueError(f'Shared memory {name} could not be initialized!.')

            _size = self.shm_manager.shm_size(_shm_handler)

            if _size >= alloc_size:
                LOGGER.debug(f'Shared memory {name} already registered!')
                shm_handler = _shm_handler
            else:
                LOGGER.info(f'Shared memory {name} with size {_size} already registered! Expected size={alloc_size}. Unlinked and updated the shared memory.')
                self.shm_manager.shm_unregister(shm_key)
                _shm_handler = self.shm_manager.shm_register(shm_key, alloc_size)

                if _shm_handler is None:
                    raise ValueError(f'Shared memory {name} could not be initialized!')

                shm_handler = _shm_handler

        buffer = self._get_buffer(shm_handler=shm_handler, size=alloc_size)

        if init_value is not None:
            buffer[:len(init_value)] = init_value
            # self._set_buffer(shm_handler=shm_handler, data=init_value, buffer_size=alloc_size, override=True)

        return MemoryBuffer(name=name, shm_handler=shm_handler, buffer=buffer, size=buffer_size)

    def get_buffer(self, name: str, size: int = None) -> MemoryBuffer | None:
        shm_handler = self._get_shm_handler(name=name)

        if shm_handler is None:
            return None

        buffer = self._get_buffer(shm_handler=shm_handler, size=size)
        return MemoryBuffer(name=name, shm_handler=shm_handler, buffer=buffer, size=len(buffer))

    def set_buffer(self, name: str, value: bytes):
        shm_handler = self._get_shm_handler(name=name)
        self._set_buffer(shm_handler=shm_handler, data=value)

    def close_buffer(self, name: str) -> int:
        res = self.shm_manager.shm_unregister(name.encode(self.encoding))

        if not res:
            LOGGER.debug(f'Failed to close shared memory {name}!')
            return -1

        LOGGER.debug(f'Shared memory {name} closed successfully!')
        return 0

    def unlink_buffer(self, to_unlink: str | MemoryBuffer = None, /, name: str = None, buffer: MemoryBuffer = None) -> int:
        if name is None and buffer is None:
            if isinstance(to_unlink, MemoryBuffer):
                name = to_unlink.name
            elif isinstance(to_unlink, str):
                name = to_unlink
            else:
                raise TypeError(f'Invalid buffer type {to_unlink}!')
        elif name is None:
            name = buffer.name

        LOGGER.debug(f'Can not unlink buffer in {os.name} platform!')
        return self.close_buffer(name)


class CachedMemoryCore(SharedMemoryCore):
    def __init__(self, prefix: str, **kwargs):
        super().__init__(**kwargs)

        # if not prefix.startswith('Global\\'):
        #     raise ValueError('Prefix must start with "Global\\" so that shared memory is accessible from all child processes!')
        # elif '\\' in prefix[7:]:
        #     raise ValueError('Prefix must not contain "\" except the characters in prefix!')

        self.prefix = prefix
        self.shm_registered: dict[str, MemoryBuffer] = {}

    def __del__(self):
        self.unlink()

    @classmethod
    def cache_name(cls, name: str) -> str:
        return f'{name}._cache_size'

    def init_buffer(self, name: str, buffer_size: int = None, init_value: bytes = None) -> MemoryBuffer:
        if name.startswith(self.prefix):
            real_name = name
        else:
            real_name = f'{self.prefix}.{name}'

        buffer = super().init_buffer(name=real_name, buffer_size=buffer_size, init_value=init_value)

        self.shm_registered[real_name] = buffer
        return buffer

    def get_buffer(self, name: str, size: int = None) -> MemoryBuffer | None:
        if name.startswith(self.prefix):
            real_name = name
        else:
            real_name = f'{self.prefix}.{name}'

        buffer = super().get_buffer(name=real_name, size=size)

        if buffer is not None:
            self.shm_registered[real_name] = buffer

        return buffer

    def set_buffer(self, name: str, value: bytes):
        if name.startswith(self.prefix):
            real_name = name
        else:
            real_name = f'{self.prefix}.{name}'

        return super().set_buffer(name=real_name, value=value)

    def close_buffer(self, name: str) -> int:
        if name.startswith(self.prefix):
            real_name = name
        else:
            real_name = f'{self.prefix}.{name}'

        return super().close_buffer(name=real_name)

    def unlink_buffer(self, to_unlink: str | MemoryBuffer = None, /, name: str = None, buffer: MemoryBuffer = None):
        if name is None and buffer is None:
            if isinstance(to_unlink, MemoryBuffer):
                name = to_unlink.name
            elif isinstance(to_unlink, str):
                name = to_unlink
            else:
                raise TypeError(f'Invalid buffer type {to_unlink}!')
        elif name is None:
            name = buffer.name

        if name.startswith(self.prefix):
            real_name = name
        else:
            real_name = f'{self.prefix}.{name}'

        res = super().unlink_buffer(name=real_name)
        self.shm_registered.pop(real_name, None)
        return res

    def close(self):
        for name in list(self.shm_registered):
            LOGGER.debug(f'{self.prefix} {self.__class__.__name__} closed "{name}" shared memory.')
            self.close_buffer(name=name)

        self.shm_registered.clear()

    def unlink(self):
        for name in list(self.shm_registered):
            LOGGER.debug(f'{self.prefix} {self.__class__.__name__} unlinked "{name}" shared memory.')
            self.close_buffer(name=name)
            self.unlink_buffer(name=name)

        self.shm_registered.clear()


class SyncMemoryCore(CachedMemoryCore):
    def __init__(self, prefix: str = None, dummy: bool = False, **kwargs):
        if prefix is None:
            prefix = f'/{uuid.uuid4().hex}'

        super().__init__(prefix=prefix, **kwargs)
        self.dummy: bool = dummy
        self.storage: dict[str, SyncTemplateBase] = {}

    def __reduce__(self):
        return self.__class__.from_json, (self.to_json(),)

    def register(self, *args, name: str, dtype: str | SyncTypes, use_cache: bool = True, **kwargs):
        if isinstance(dtype, SyncTypes):
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
                case SyncTypes.NamedVector.value:
                    sync_storage = NamedVector(manager=self, name=name, *args, **kwargs)
                case SyncTypes.Vector.value:
                    sync_storage = Vector(manager=self, name=name, *args, **kwargs)
                case SyncTypes.Deque.value:
                    sync_storage = Deque(manager=self, name=name, *args, **kwargs)
                case SyncTypes.IntValue.value:
                    sync_storage = IntValue(manager=self, name=name, *args, **kwargs)
                case SyncTypes.FloatValue.value:
                    sync_storage = FloatValue(manager=self, name=name, *args, **kwargs)
                case _:
                    raise ValueError(f'Invalid dtype {dtype}.')

        self.storage[name] = sync_storage
        return sync_storage

    def unregister(self, name: str):
        sync_storage = self.storage.pop(name, None)

        # if sync_storage is not None:
        #     sync_storage.unlink()

        return sync_storage

    def register_dummy(self, *args, name: str, dtype: str | SyncTypes, **kwargs):
        if isinstance(dtype, SyncTypes):
            dtype = dtype.value

        if not self.dummy:
            LOGGER.warning('Using register_dummy may override the synchronized storage. This function is for testing only.')

        match dtype:
            case SyncTypes.NamedVector.value:
                sync_storage = NamedVectorDummy(manager=self, name=name, *args, **kwargs)
            case SyncTypes.Vector.value:
                sync_storage = VectorDummy(manager=self, name=name, *args, **kwargs)
            case SyncTypes.Deque.value:
                sync_storage = DequeDummy(manager=self, name=name, *args, **kwargs)
            case SyncTypes.IntValue.value | SyncTypes.FloatValue.value:
                sync_storage = ValueDummy(manager=self, name=name, *args, **kwargs)
            case _:
                raise ValueError(f'Invalid dtype {dtype}.')

        self.storage[name] = sync_storage
        return sync_storage

    def close(self):
        LOGGER.debug(f'{self.prefix} memory core closed {len(self.storage)} storage entries, {len(self.shm_registered)} cache entries!')
        for name, sync_storage in list(self.storage.items()):
            LOGGER.debug(f'{self.prefix} memory core closing {sync_storage.__class__.__name__} {name} buffer.')
            sync_storage.close()

        super().close()

        self.storage.clear()
        self.shm_registered.clear()

    def unlink(self):
        LOGGER.debug(f'{self.prefix} memory core unlinked {len(self.storage)} storage entries, {len(self.shm_registered)} cache entries!')
        for name, sync_storage in list(self.storage.items()):
            LOGGER.debug(f'{self.prefix} memory core unlinked {sync_storage.__class__.__name__} {name} buffer.')
            sync_storage.unlink()

        super().unlink()

        self.storage.clear()
        self.shm_registered.clear()

    def clear(self):
        self.storage.clear()
        self.shm_registered.clear()

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

        self.storage = {name: SyncTemplate.from_json(manager=self, json_message=data) for name, data in json_dict['storage'].items()}

        # self.from_shm(override=True)
        return self

    @property
    def is_sync(self) -> bool:
        for sync_storage in self.storage.values():
            is_sync = sync_storage.is_sync

            if not is_sync:
                return False

        return True


class SyncTemplate(SyncTemplateBase, metaclass=abc.ABCMeta):
    def __init__(self, manager: SyncMemoryCore, name: str, **kwargs):
        super().__init__(
            name=name,
            manager=manager,
            **kwargs
        )

        self.desync_warning_limit = kwargs.get('desync_warning_limit', None)

    def close(self):
        try:
            self._manager: SyncMemoryCore
            super().close()
        except FileNotFoundError as _:
            self.clear(silent=True)

    def unlink(self):
        try:
            self._manager: SyncMemoryCore
            super().unlink()
        except FileNotFoundError as _:
            self.clear(silent=True)

    def new_ver(self, desync_warning_limit: int = None):
        self._ver_local: bytes = uuid.uuid4().bytes

        self.desync_counter += 1

        desync_warning_limit = desync_warning_limit if desync_warning_limit is not None else self.desync_warning_limit

        if desync_warning_limit is not None and self.desync_counter >= desync_warning_limit > 0:
            LOGGER.error(f'{self} de-synchronize too many times, check the code!')


class NamedVector(SyncTemplate):
    def __init__(self, *args, manager: SyncMemoryCore, name: str, **kwargs):
        data = dict(*args, **kwargs)

        super().__init__(name=name, manager=manager, ver_shm=manager.init_buffer(name=f'{name}.ver_code', buffer_size=16))
        self._dtype = np.double
        self._keys: list[str] = list(data.keys())
        self._idx: dict[str, int] = {key: idx for idx, key in enumerate(self._keys)}
        self._values: np.ndarray = np.array(list(data.values()), dtype=self._dtype)

        self._shm_keys: MemoryBuffer | None = None
        self._shm_values: MemoryBuffer | None = None

        self._register_buffer_attribute('_shm_keys')
        self._register_buffer_attribute('_shm_values')

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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
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

        # no need to sync an empty dict
        if not self:
            LOGGER.warning(f'Sync empty {self.__class__.__name__} {self.name} is not advised, to free up memory, use unlink instead.')
            LOGGER.warning(f'{self.name} has no data, and will not be synchronized,')
            return

        if not is_sync or self._shm_keys is None or self._shm_values is None:
            self._shm_keys = self._manager.init_buffer(name=self.name_keys, init_value=self._manager.convertor.str_vector_to_bytes(self._keys))
            self._shm_values = self._manager.init_buffer(name=self.name_values, init_value=self._manager.convertor.array_to_bytes(self._values[:len(self)]))
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        if self.empty_shm:
            return

        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_keys is None or self._shm_values is None:
            self._shm_keys = self._manager.get_buffer(name=self.name_keys)
            self._shm_values = self._manager.get_buffer(name=self.name_values)

            if self._shm_keys is None or self._shm_values is None:
                LOGGER.info(f'No shm found for {self.name}, {self.__class__.__name__} not synchronized.')
                return

            self._keys = self._manager.convertor.bytes_to_str_vector(data=self._shm_keys.to_bytes().rstrip(b'\x00'))
            self._idx = {key: idx for idx, key in enumerate(self._keys)}
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
            self.get_ver()
            return

    def __setitem__(self, key: str, value: float):
        if key not in self._idx:
            self._idx[key] = _idx = len(self._keys)
            self._keys.append(key)
            if len(self._values) == _idx:
                self._values = np.append(self._values, value)
            elif len(self._values) > _idx:
                self._values[_idx] = value
            else:
                raise RuntimeError('Named vector not synchronized!')

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

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>{dict(self.items())}'

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

    @property
    def name_keys(self) -> str:
        return f'{self._name}.keys'

    @property
    def name_values(self) -> str:
        return f'{self._name}.values'


class Vector(SyncTemplate):
    def __init__(self, *args, manager: SyncMemoryCore, name: str, **kwargs):
        data = list(*args, **kwargs)

        super().__init__(name=name, manager=manager, ver_shm=manager.init_buffer(name=f'{name}.ver_code', buffer_size=16))
        self._dtype = np.double
        self._values = np.array(data, dtype=self._dtype)

        self._shm_values: MemoryBuffer | None = None

        self._register_buffer_attribute('_shm_values')

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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
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
            self._shm_values = self._manager.init_buffer(name=self.name, init_value=self._manager.convertor.array_to_bytes(self._values))
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        if self.empty_shm:
            return

        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_values is None:
            self._shm_values = self._manager.get_buffer(name=self.name)

            if self._shm_values is None:
                LOGGER.info(f'No shm found for {self.name}, {self.__class__.__name__} not synchronized.')
                return

            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
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

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>{self._values.tolist()}'

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


class Deque(SyncTemplate):
    def __init__(self, *args, manager: SyncMemoryCore, name: str, maxlen: int, **kwargs):
        data = list(*args, **kwargs)

        super().__init__(name=name, manager=manager, ver_shm=manager.init_buffer(name=f'{name}.ver_code', buffer_size=16))
        self._maxlen = maxlen
        self._dtype = np.double
        self._values = np.zeros(shape=maxlen, dtype=self._dtype)
        self._length = min(maxlen, len(data))

        if self._length:
            self._values[-self._length:] = data[-self._length:]

        self._shm_values: MemoryBuffer | None = None
        self._shm_length: MemoryBuffer | None = None

        self._register_buffer_attribute('_shm_values')
        self._register_buffer_attribute('_shm_length')

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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
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
            self._shm_values = self._manager.init_buffer(name=self.name, init_value=self._manager.convertor.array_to_bytes(self._values))
            self._shm_length = self._manager.init_buffer(name=self.name_length, init_value=self._manager.convertor.int_to_bytes(self._length))
            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        if self.empty_shm:
            return

        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm_values is None or self._shm_length is None:
            self._shm_values = self._manager.get_buffer(name=self.name)
            self._shm_length = self._manager.get_buffer(name=self.name_length)

            if self._shm_values is None or self._shm_length is None:
                LOGGER.info(f'No shm found for {self.name}, {self.__class__.__name__} not synchronized.')
                return

            self._values = np.ndarray(shape=(-1,), dtype=self._dtype, buffer=self._shm_values.buffer)
            self._length = int.from_bytes(self._shm_length.buffer)
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

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>{self.data.tolist()}'

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
        # if not silent:
        #     LOGGER.warning(f'{self.__class__.__name__} should not be clear. If it is to free the resource, use unlink().')

        if self.is_sync:
            self._values[:] = np.zeros(shape=self._maxlen, dtype=self._dtype)
            self.length = 0
        else:
            self._values = np.zeros(shape=self._maxlen, dtype=self._dtype)
            self.new_ver()
            self._length = 0

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
            return int.from_bytes(self._shm_length.buffer)

    @length.setter
    def length(self, value: int):
        self._length = value

        if self._shm_length:
            self._shm_length.buffer[:] = self._length.to_bytes(length=self._shm_length.size)


class IntValue(SyncTemplate):
    def __init__(self, value: int, manager: SyncMemoryCore, name: str, size: int = 8):
        super().__init__(name=name, manager=manager, ver_shm=manager.init_buffer(name=f'{name}.ver_code', buffer_size=16))
        self._size = size
        self._value: int = value

        self._shm: MemoryBuffer | None = None

        self._register_buffer_attribute('_shm')

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>(value={self.value})'

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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
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
            self._shm = self._manager.init_buffer(name=self.name, init_value=self._manager.convertor.int_to_bytes(self._value))
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        if self.empty_shm:
            return

        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm is None:
            self._shm = self._manager.get_buffer(name=self.name)

            if self._shm is None:
                LOGGER.info(f'No shm found for {self.name}, {self.__class__.__name__} not synchronized.')
                return

            self._value = self._manager.convertor.bytes_to_int(self._shm.buffer)
            self.get_ver()
            return

    def clear(self, silent: bool = False):
        pass

    @property
    def value(self):
        if self._shm is not None:
            self._value = self._manager.convertor.bytes_to_int(self._shm)

        return self._value

    @value.setter
    def value(self, new_value: int):
        self._value = new_value
        self.to_shm(override=True)


class FloatValue(SyncTemplate):
    def __init__(self, value: float, manager: SyncMemoryCore, name: str, size: int = 8):
        super().__init__(name=name, manager=manager, ver_shm=manager.init_buffer(name=f'{name}.ver_code', buffer_size=16))

        if size != 8:
            LOGGER.warning(f'numpy double byte size should be 8, not {size} at most of the platforms. Use with caution.')

        self._size = size
        self._value = np.array([value], dtype=np.double)

        self._shm: MemoryBuffer | None = None

        self._register_buffer_attribute('_shm')

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.name}>(value={self.value})'

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
    def from_json(cls, manager: SyncMemoryCore, json_message: str | bytes | bytearray | dict, **kwargs) -> Self:
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
            self._shm = self._manager.init_buffer(name=self.name, init_value=self._value.tobytes())
            self._value = np.ndarray(shape=(1,), dtype=np.double, buffer=self._shm.buffer)
            self.set_ver()
            return

    def from_shm(self, override: bool = False):
        if self.empty_shm:
            return

        is_sync = self.is_sync and not override

        if (not is_sync) or self._shm is None:
            self._shm = self._manager.get_buffer(name=self.name)

            if self._shm is None:
                LOGGER.info(f'No shm found for {self.name}, {self.__class__.__name__} not synchronized.')
                return

            self._value = np.ndarray(shape=(1,), dtype=np.double, buffer=self._shm.buffer)
            self.get_ver()
            return

    def clear(self, silent: bool = False):
        pass

    @property
    def value(self):
        return self._value[0]

    @value.setter
    def value(self, new_value: int):
        self._value[0] = new_value


__all__ = ['SharedMemoryCore', 'CachedMemoryCore', 'SyncMemoryCore',
           'SyncTemplate', 'Vector', 'NamedVector', 'Deque', 'IntValue', 'FloatValue']
