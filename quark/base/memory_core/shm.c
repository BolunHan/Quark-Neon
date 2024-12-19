#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Function to register a shared memory segment using shm_open
int shm_register(const char *key, size_t size) {
    int fd = shm_open(key, O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd == -1) {
        return -1; // Return -1 if shm_open fails
    }

    // Set the size of the shared memory segment
    if (ftruncate(fd, size) == -1) {
        close(fd);
        shm_unlink(key);
        return -1; // Return -1 if ftruncate fails
    }

    return fd;
}

// Function to get the size of a shared memory segment
ssize_t shm_size(int fd) {
    struct stat shm_stat;
    if (fstat(fd, &shm_stat) == -1) {
        return -1; // Return -1 if fstat fails
    }
    return shm_stat.st_size;
}

// Function to get an already registered shared memory segment
int shm_get(const char *key) {
    int fd = shm_open(key, O_RDWR, 0666);
    if (fd == -1) {
        return -1; // Return -1 if shm_open fails
    }
    return fd;
}

// Function to get a pointer to the shared memory buffer
void* shm_buffer(int fd, size_t size) {
    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        return NULL; // Return NULL if mmap fails
    }
    return addr; // Return the pointer to the shared memory
}

// Function to close the file descriptor of a shared memory segment
int shm_close(int fd) {
    if (close(fd) == -1) {
        return -1; // Return -1 if close fails
    }
    return 0; // Return 0 on success
}

// Function to unregister a shared memory segment using shm_unlink
int shm_unregister(const char *key) {
    if (shm_unlink(key) == -1) {
        return -1; // Return -1 if shm_unlink fails
    }
    return 0; // Return 0 on success
}

// Function to write bytes to shared memory
int shm_bytes_write(int fd, const void *data, size_t size) {
    void *shmaddr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (shmaddr == MAP_FAILED) {
        return -1; // Return -1 if mmap fails
    }

    memcpy(shmaddr, data, size);

    if (munmap(shmaddr, size) == -1) {
        return -1; // Return -1 if munmap fails
    }

    return 0; // Return 0 on success
}

// Function to read bytes from shared memory
int shm_bytes_read(int fd, void *buffer, size_t size) {
    void *shmaddr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (shmaddr == MAP_FAILED) {
        return -1; // Return -1 if mmap fails
    }

    memcpy(buffer, shmaddr, size);

    if (munmap(shmaddr, size) == -1) {
        return -1; // Return -1 if munmap fails
    }

    return 0; // Return 0 on success
}

// Function to write an integer to shared memory
int shm_int_write(int fd, int value) {
    return shm_bytes_write(fd, &value, sizeof(int));
}

// Function to read an integer from shared memory into a buffer
int shm_int_read(int fd, int *buffer) {
    return shm_bytes_read(fd, buffer, sizeof(int));
}

// Function to write a double to shared memory
int shm_dbl_write(int fd, double value) {
    return shm_bytes_write(fd, &value, sizeof(double));
}

// Function to read a double from shared memory into a buffer
int shm_dbl_read(int fd, double *buffer) {
    return shm_bytes_read(fd, buffer, sizeof(double));
}