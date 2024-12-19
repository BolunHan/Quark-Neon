__package__ = 'quark.base.memory_core'

import ctypes
import pathlib
import logging
import pickle
import sys
import uuid

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
stdout_handler.setLevel(logging.INFO)
LOGGER.addHandler(stdout_handler)
stderr_handler = logging.StreamHandler(stream=sys.stderr)
stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
stderr_handler.setLevel(logging.INFO)
LOGGER.addHandler(stderr_handler)

# Load the shared library
here = pathlib.Path(__file__).parent.absolute()
# shm_lib = here.joinpath('shm.so')
shm_lib = pathlib.Path(r'/home/bolun/.pyenv/versions/venv_312/lib/python3.12/site-packages/quark/base/memory_core/shm.cpython-312-x86_64-linux-gnu.so')
memory_manager = ctypes.CDLL(str(shm_lib))

# Function signatures
memory_manager.shm_register.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
memory_manager.shm_register.restype = ctypes.c_int

memory_manager.shm_size.argtypes = [ctypes.c_int]
memory_manager.shm_size.restype = ctypes.c_ssize_t

memory_manager.shm_get.argtypes = [ctypes.c_char_p]
memory_manager.shm_get.restype = ctypes.c_int

memory_manager.shm_unregister.argtypes = [ctypes.c_char_p]
memory_manager.shm_unregister.restype = ctypes.c_int

memory_manager.shm_bytes_write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
memory_manager.shm_bytes_write.restype = ctypes.c_int

memory_manager.shm_bytes_read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
memory_manager.shm_bytes_read.restype = ctypes.c_int

memory_manager.shm_int_write.argtypes = [ctypes.c_int, ctypes.c_int]
memory_manager.shm_int_write.restype = ctypes.c_int

memory_manager.shm_int_read.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
memory_manager.shm_int_read.restype = ctypes.c_int

memory_manager.shm_dbl_write.argtypes = [ctypes.c_int, ctypes.c_double]
memory_manager.shm_dbl_write.restype = ctypes.c_int

memory_manager.shm_dbl_read.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
memory_manager.shm_dbl_read.restype = ctypes.c_int

memory_manager.shm_buffer.argtypes = [ctypes.c_int, ctypes.c_size_t]
memory_manager.shm_buffer.restype = ctypes.c_void_p


def _test_init_shm():
    key = b'/shm_test'
    size = 1024

    # test registration
    fd = memory_manager.shm_register(key, size)
    if fd == -1:
        LOGGER.error(f'Failed to register shm with {key}!')
    else:
        LOGGER.info(f'registered shm with {key}!')
    fd = memory_manager.shm_register(key, 1024)
    if fd == -1:
        LOGGER.info(f'Failed to register shm {key} as expected!')
    else:
        LOGGER.error(f'Expected shm {key} register failure, got success instead!')

    # test get shm fd
    fd = memory_manager.shm_get(key, size)
    if fd == -1:
        LOGGER.error(f'Failed to get shm with {key}!')
    else:
        LOGGER.info(f'Got shm with {key}!')

    # test get size
    _size = memory_manager.shm_size(fd)
    if _size == -1:
        LOGGER.error(f'Failed to get shm size with {key}!')
    elif _size != size:
        LOGGER.error(f'Expected shm size {size} but got {_size}!')
    else:
        LOGGER.info(f'Got shm size {_size} with {key}!')

    # test de-registration
    fd = memory_manager.shm_unregister(key)
    if fd == -1:
        LOGGER.error(f'Failed to unregister shm {key} as expected!')
    else:
        LOGGER.info(f'Unregistered shm with {key}!')
    fd = memory_manager.shm_unregister(key)
    if fd == -1:
        LOGGER.info(f'Failed to unregister shm {key} as expected!')
    else:
        LOGGER.error(f'Expected shm {key} unregister failure, got success instead!')


def _test_shm_access():
    key = f"/shm_example/{uuid.uuid4().hex[:5]}".encode('utf-8')
    key = f"/shm_example_abcd".encode('utf-8')
    size = 10240
    fd = memory_manager.shm_register(key, size)

    if fd == -1:
        LOGGER.error(f'Failed to register shm with {key}! This might comes from a previous shm not unlinked successfully!')
        memory_manager.shm_unregister(key)
        # key = f"/shm_example/{uuid.uuid4()}".encode('utf-8')
        fd = memory_manager.shm_register(key, size)

    if fd == -1:
        LOGGER.error(f'Failed to register shm with {key}!')

    # test int
    int_value = 42
    LOGGER.info(f'Sending test int to shm {int_value}!')
    if memory_manager.shm_int_write(fd, int_value) == -1:
        LOGGER.error("Error: Could not write to shared memory.")
    new_int_buffer = ctypes.c_int()
    if memory_manager.shm_int_read(fd, ctypes.byref(new_int_buffer)) == -1:
        LOGGER.error("Error: Could not read from shared memory.")
    else:
        LOGGER.info(f'Read int from shm {new_int_buffer.value}')

    # test float
    float_value = np.pi
    LOGGER.info(f'Sending test float to shm {float_value}!')
    memory_manager.shm_dbl_write(fd, float_value)
    new_dbl_buffer = ctypes.c_double()
    memory_manager.shm_dbl_read(fd, ctypes.byref(new_dbl_buffer))
    LOGGER.info(f'Read float from shm {new_dbl_buffer.value}')

    # test string
    test_str = f'hello world\n' + '\n'.join([str(uuid.uuid4()) for _ in range(1)])
    LOGGER.info(f'Sending test str to shm...')
    LOGGER.debug(f'test str:\n"{test_str}"!')
    str_buffer = ctypes.create_string_buffer(test_str.encode('utf-8'))
    memory_manager.shm_bytes_write(fd, ctypes.byref(str_buffer), ctypes.sizeof(str_buffer))
    new_str_buffer = ctypes.create_string_buffer(size)
    memory_manager.shm_bytes_read(fd, ctypes.byref(new_str_buffer), ctypes.sizeof(new_str_buffer))
    new_str = new_str_buffer.value.decode("utf-8").strip()
    LOGGER.debug(f'Read str from shm "{new_str}"')
    if new_str != test_str:
        LOGGER.error('string from shm does not match!')
    else:
        LOGGER.info('string from shm matches!')

    # test bytes
    test_bytes_0 = uuid.uuid4().bytes
    test_bytes_1 = uuid.uuid4().bytes
    local_byte_buffer = (ctypes.c_char * 16)()
    memory_manager.shm_bytes_write(fd, test_bytes_0, 16)
    memory_manager.shm_bytes_read(fd, ctypes.byref(local_byte_buffer), 16)
    assert local_byte_buffer.value == test_bytes_0
    memory_manager.shm_bytes_write(fd, test_bytes_1, 16)
    memory_manager.shm_bytes_read(fd, ctypes.byref(local_byte_buffer), 16)
    assert local_byte_buffer.value == test_bytes_1

    # test vector
    test_vector = [np.pi, np.e, np.nan, 0, -np.inf]
    arr = np.array(test_vector)
    test_bytes = arr.tobytes()
    # test_bytes = 'hello world'.encode('utf-8')
    c_data = ctypes.cast(ctypes.c_char_p(test_bytes), ctypes.POINTER(ctypes.c_byte))
    # c_data = ctypes.py_object(test_bytes)
    memory_manager.shm_bytes_write(fd, c_data, arr.nbytes)
    # memory_manager.shm_bytes_write(fd, ctypes.byref(c_data), 8)
    ptr = memory_manager.shm_buffer(fd, size)
    if not ptr:
        raise MemoryError("Failed to mmap the shared memory segment.")

    buf = (ctypes.c_byte * arr.nbytes).from_address(ptr)
    # m = memoryview(buf)
    LOGGER.info(f'the vector from the memory buffer is {np.ndarray(shape=(-1,), buffer=buf).tolist()}')

    # test pickle
    # obj = f'hello world\n'
    obj = f'hello world\n' + '\n'.join([str(uuid.uuid4()) for _ in range(100)])
    test_bytes = pickle.dumps(obj)
    # c_data = ctypes.cast(ctypes.c_char_p(test_bytes), ctypes.POINTER(ctypes.c_byte))
    # c_data = ctypes.create_string_buffer(test_bytes)
    # memory_manager.shm_bytes_write(fd, c_data, len(test_bytes))
    # memory_manager.shm_bytes_write(fd, ctypes.byref(c_data), len(test_bytes))
    memory_manager.shm_bytes_write(fd, test_bytes, size)
    ptr = memory_manager.shm_buffer(fd, len(test_bytes))
    if not ptr:
        raise MemoryError("Failed to mmap the shared memory segment.")

    buf = (ctypes.c_byte * len(test_bytes)).from_address(ptr)
    # m = memoryview(buf)
    LOGGER.info(f'the obj from the memory buffer is {pickle.loads(buf)}')


def _test_shm_np():
    original_array = np.array([np.pi, np.e, np.nan, 0, -np.inf])
    key = b'/shm_np'
    size = original_array.nbytes

    # test registration
    fd = memory_manager.shm_register(key, size)
    if fd == -1:
        memory_manager.shm_unregister(key)
        fd = memory_manager.shm_register(key, size)
    LOGGER.info(f'Sending numpy array {original_array} to shm!')
    memory_manager.shm_bytes_write(fd, bytes(original_array.data), size)
    addr = memory_manager.shm_buffer(fd, size)
    ctypes.pythonapi.PyMemoryView_FromMemory.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
    ctypes.pythonapi.PyMemoryView_FromMemory.restype = ctypes.py_object

    buffer = ctypes.pythonapi.PyMemoryView_FromMemory(ctypes.c_void_p(addr), ctypes.c_ssize_t(size), ctypes.c_int(0x200))
    shm_arr_0 = np.ndarray(shape=(-1,), dtype=np.double, buffer=memoryview(buffer))
    memory_manager.shm_get(key)
    addr_2 = memory_manager.shm_buffer(fd, 24)
    buffer_2 = ctypes.pythonapi.PyMemoryView_FromMemory(ctypes.c_void_p(addr_2), ctypes.c_ssize_t(24), ctypes.c_int(0x200))
    shm_arr_1 = np.ndarray(shape=(-1,), dtype=np.double, buffer=buffer_2)

    LOGGER.info(f'Get numpy array {shm_arr_0} from shm!')
    shm_arr_0[0] = 1
    LOGGER.info(f'updating numpy array to {shm_arr_0} from shm!')
    LOGGER.info('expecting to see new array with same memory buffer updated simultaneously!')
    LOGGER.info(shm_arr_1)
    memory_manager.shm_unregister(key)


def _test_shm_manager():
    from .shm_manager_posix import SyncMemoryCore
    manager = SyncMemoryCore(prefix='/manager.test')

    # step 0: test int, float, bytes read and write access
    int_set = 42
    buffer = manager.init_buffer(name='test_int', init_value=int_set.to_bytes(8))
    LOGGER.info(f'{buffer.to_bytes()}')
    LOGGER.info(f'Sending int value {int_set} to the shm...')
    int_get = manager.get_int(name='test_int')
    if int_get == int_get:
        LOGGER.info('Getting int value from shm successfully!')
    else:
        LOGGER.error('Getting int value from shm failed.')

    dbl_set = np.pi
    manager.set_double(name='test_dbl', value=dbl_set)
    LOGGER.info(f'Sending dbl value {dbl_set} to the shm...')
    dbl_get = manager.get_double(name='test_dbl')
    if dbl_get == dbl_get:
        LOGGER.info('Getting dbl value from shm successfully!')
    else:
        LOGGER.error('Getting dbl value from shm failed.')

    bytes_set = pickle.dumps({'a': 1, 23: 'n', 'd': None})
    manager.init_buffer(name='test_bytes', init_value=bytes_set)
    LOGGER.info(f'Sending bytes value {bytes_set} to the shm...')
    bytes_get = manager.get_bytes(name='test_bytes')
    if bytes_get == bytes_get:
        LOGGER.info('Getting bytes value from shm successfully!')
    else:
        LOGGER.error('Getting bytes value from shm failed.')

    manager.unlink()


def _test_named_vector():
    from .shm_manager_posix import SyncMemoryCore, SyncTypes
    manager = SyncMemoryCore(prefix='/manager.test')

    named_vector_0 = manager.register(dtype=SyncTypes.NamedVector, name='dict')
    named_vector_1 = manager.register(dtype=SyncTypes.NamedVector, name='dict', use_cache=False)

    named_vector_0['a'] = 1
    named_vector_0['b'] = 2
    named_vector_0.to_shm()
    named_vector_1.from_shm()

    LOGGER.info(named_vector_0)
    LOGGER.info(named_vector_1)

    named_vector_0['a'] = 2
    named_vector_1['b'] = 3

    LOGGER.info(named_vector_0)
    LOGGER.info(named_vector_1)

    named_vector_0.close()
    named_vector_1.unlink()


def main():
    _test_init_shm()
    _test_shm_access()
    _test_shm_np()
    _test_shm_manager()
    _test_named_vector()


if __name__ == '__main__':
    main()
