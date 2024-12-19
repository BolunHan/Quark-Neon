__package__ = 'quark.base.memory_core'

import ctypes
import mmap
import os
import uuid

lib_path = os.path.abspath(r"C:\Users\Bolun\Projects\Quark\quark\base\memory_core\shm.dll")
# Load the shared library
lib = ctypes.CDLL(lib_path)

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


def _test_register_shm():
    # Example usage
    key = f"shm.test.{uuid.uuid4().hex[:5]}".encode('utf-8')
    size = mmap.PAGESIZE
    handle = lib.shm_register(key, size)

    if handle is None:
        raise ValueError(f'shm {key} register failed!')
    print(f'shm {key} register successful!')

    handle_2 = lib.shm_get(key)
    if handle_2 is None:
        raise RuntimeError(f'Failed to get shm handler {key}!')
    print(f'shm {key} get successful!')

    res = lib.shm_unregister(key)
    if res == -1:
        raise ValueError(f'shm {key} unregister failed!')
    print(f'shm {key} unregister successful!')

    handle = lib.shm_register(key, size)
    if handle is None:
        raise ValueError(f'shm {key} register failed!')
    print(f'shm {key} re-register successful!')

    handle_f = lib.shm_register(key, size)
    if handle_f is not None:
        raise RuntimeError(f'shm {key} should not be registered, but still reported a success!')
    print(f'shm {key} re-register failed as expected!')

    res = lib.shm_unregister(key)
    if not res:
        raise ValueError(f'shm {key} unregister failed!')
    print(f'shm {key} unregister successful!')

    handle = lib.shm_register(key, size)
    if handle is None:
        raise ValueError(f'shm {key} register failed!')
    print(f'shm {key} re-register successful!')

    res = lib.shm_close(handle)
    if res == -1:
        raise ValueError(f'handler {handle} close failed!')
    print(f'handler {handle} close successful!')

    res = lib.shm_unregister(key)
    if res:
        raise ValueError(f'shm {key} unregister expected to fail, but still reported a success!')
    print(f'shm {key} unregister failed as expected!')

    res = lib.shm_close(handle)
    if res == 0:
        raise ValueError(f'handler {handle_2} close expected to fail, but still reported a success!')
    print(f'handler {handle_2} close failed as expected!')

    handle = lib.shm_register(key, size)
    if handle is None:
        raise ValueError(f'shm {key} register failed!')
    print(f'shm {key} re-register successful!')

    # res = lib.shm_close(handle)
    res = lib.shm_unregister(key)
    if res == -1:
        raise ValueError(f'shm {key} unregister failed!')
    print(f'shm {key} unregister successful!')


def _test_shm_wr():
    # key = f"shm.test.{uuid.uuid4().hex[:5]}".encode('utf-8')
    key = b"/AsyncMonitorManager.39c111c3c8c544eb8069f4a2956542e2.monitor_value"
    size = 4098  # in windows platform, this should lead to an allocated size of 4096 * 2
    alloc_page = size // mmap.PAGESIZE
    alloc_size = alloc_page * mmap.PAGESIZE
    shm_1 = lib.shm_register(key, alloc_size)

    if shm_1 is None:
        raise ValueError(f'shm {key} register failed!')
    print(f'shm {key} register successful!')

    ptr_to_buffer = ctypes.pythonapi.PyMemoryView_FromMemory
    ptr_to_buffer.argtypes = (ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int)
    ptr_to_buffer.restype = ctypes.py_object

    addr_1 = lib.shm_buffer(shm_1, alloc_size)
    buffer_1: memoryview = ptr_to_buffer(ctypes.c_void_p(addr_1), ctypes.c_ssize_t(alloc_size), ctypes.c_int(0x200))
    buffer_1[:] = int(182736).to_bytes(length=alloc_size, byteorder='big')

    shm_2 = lib.shm_get(key)
    size_2 = lib.shm_size(shm_2)
    assert size_2 == alloc_size
    if shm_2 is None:
        raise ValueError(f'shm {key} get failed!')

    addr_2 = lib.shm_buffer(shm_2, size_2)
    buffer_2 = ptr_to_buffer(ctypes.c_void_p(addr_2), ctypes.c_ssize_t(size_2), ctypes.c_int(0x200))

    bytes(buffer_2)
    assert bytes(buffer_2) == bytes(buffer_1)


def _test_named_vector():
    from .shm_manager_nt import SyncMemoryCore, SyncTypes, LOGGER
    manager = SyncMemoryCore(prefix='test')

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

    named_vector_0['c'] = 4
    named_vector_0.to_shm()
    named_vector_1.from_shm()

    LOGGER.info(named_vector_0)
    LOGGER.info(named_vector_1)

    named_vector_0.close()
    named_vector_1.unlink()


def main():
    _test_register_shm()
    _test_shm_wr()
    _test_named_vector()


if __name__ == '__main__':
    main()
