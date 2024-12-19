#include <windows.h>
#include <stdio.h>

// Export declaration macro
#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif

// Structure to associate each handle with its key
typedef struct {
    HANDLE handle;
    char key[256];
} HandleEntry;

// Global array to track all handles with associated keys
#define MAX_HANDLES 1024
HandleEntry g_handles[MAX_HANDLES];
int g_handle_count = 0;

// Helper function to track a handle with its key
void track_handle(HANDLE hMapFile, const char *key) {
    if (g_handle_count < MAX_HANDLES) {
        g_handles[g_handle_count].handle = hMapFile;
        strncpy(g_handles[g_handle_count].key, key, 256);
        g_handle_count++;
    }
}

// Helper function to untrack (remove) a handle by its handle value
void untrack_handle(HANDLE hMapFile) {
    for (int i = 0; i < g_handle_count; i++) {
        if (g_handles[i].handle == hMapFile) {
            // Shift the remaining handles down to fill the gap
            for (int j = i; j < g_handle_count - 1; j++) {
                g_handles[j] = g_handles[j + 1];
            }

            // Clear the last handle entry to prevent dangling data
            g_handles[g_handle_count - 1].handle = NULL;
            g_handles[g_handle_count - 1].key[0] = '\0';

            g_handle_count--; // Decrease the global handle count
            break;
        }
    }
}

// Function to register a shared memory segment
DLL_EXPORT HANDLE shm_register(const char *key, size_t size) {
    HANDLE hMapFile = CreateFileMappingA(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        (DWORD)size,
        key
    );

    // If CreateFileMapping fails or the key already exists
    if (hMapFile == NULL) {
        return NULL;
    }

    // Check if it already exists
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        // The shared memory already exists, so open it using OpenFileMapping
        CloseHandle(hMapFile); // Close the original handle
        return NULL;
    }

    // Track the newly created or opened handle
    track_handle(hMapFile, key);

    return hMapFile;
}

// Function to unregister (close) all handles associated with a specific key
DLL_EXPORT int shm_unregister(const char *key) {
    int closed_count = 0;

    // Iterate through all handles
    for (int i = 0; i < g_handle_count; i++) {
        if (strcmp(g_handles[i].key, key) == 0) {
            // Close the handle associated with this key
            if (CloseHandle(g_handles[i].handle)) {
                // Untrack the handle
                untrack_handle(g_handles[i].handle);
                closed_count++;

                // After untracking, reduce the loop index to recheck the next handle
                i--;
            } else {
                return -1;  // Return error if CloseHandle fails
            }
        }
    }

    // Return the count of closed handles
    return closed_count;
}

// Function to get an already registered shared memory segment
DLL_EXPORT HANDLE shm_get(const char *key) {
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        key
    );
    if (hMapFile == NULL) {
        return NULL;
    }
    track_handle(hMapFile, key);
    return hMapFile;
}

// Function to close a specific handle of a shared memory segment
DLL_EXPORT int shm_close(HANDLE hMapFile) {
    if (CloseHandle(hMapFile) == 0) {
        return -1; // Return -1 if closing the handle fails
    }

    // Use the untrack_handle function to remove the handle from the global tracking list
    untrack_handle(hMapFile);

    return 0; // Return 0 on success
}

DLL_EXPORT size_t shm_size(HANDLE hMapFile) {
    void *pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_READ,
        0,
        0,
        0
    );
    if (pBuf == NULL) {
        return -1; // Return error if mapping fails
    }

    MEMORY_BASIC_INFORMATION info;
    if (VirtualQuery(pBuf, &info, sizeof(info)) == 0) {
        UnmapViewOfFile(pBuf);
        return -1; // Return error if query fails
    }

    UnmapViewOfFile(pBuf);
    return info.RegionSize;
}

DLL_EXPORT void* shm_buffer(HANDLE hMapFile, size_t size) {
    void *addr = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size
    );
    if (addr == NULL) {
        return NULL;
    }
    return addr;
}

DLL_EXPORT int shm_bytes_write(void *shmaddr, const void *data, size_t size) {
    memcpy(shmaddr, data, size);
    return 0;
}

DLL_EXPORT int shm_bytes_read(void *shmaddr, void *buffer, size_t size) {
    memcpy(buffer, shmaddr, size);
    return 0;
}

DLL_EXPORT int shm_int_write(void *shmaddr, int value) {
    return shm_bytes_write(shmaddr, &value, sizeof(int));
}

DLL_EXPORT int shm_int_read(void *shmaddr, int *buffer) {
    return shm_bytes_read(shmaddr, buffer, sizeof(int));
}

DLL_EXPORT int shm_dbl_write(void *shmaddr, double value) {
    return shm_bytes_write(shmaddr, &value, sizeof(double));
}

DLL_EXPORT int shm_dbl_read(void *shmaddr, double *buffer) {
    return shm_bytes_read(shmaddr, buffer, sizeof(double));
}
