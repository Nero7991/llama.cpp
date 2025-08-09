#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS Memory File Format Constants
#define ATLAS_MAGIC 0x534C5441  // "ATLS" in little endian
#define ATLAS_VERSION 1
#define ATLAS_MAX_ENTRIES 65536
#define ATLAS_MAX_KEY_LENGTH 256
#define ATLAS_MAX_DATA_SIZE (1024 * 1024 * 1024)  // 1GB max per entry

// ATLAS Memory File Header Structure
typedef struct {
    uint32_t magic;          // ATLAS_MAGIC (0x534C5441)
    uint32_t version;        // File format version
    uint32_t checksum;       // CRC32 of data section
    uint32_t entry_count;    // Number of memory entries
    uint64_t created_time;   // Unix timestamp of creation
    uint64_t modified_time;  // Unix timestamp of last modification
    uint64_t total_size;     // Total size of data section
    char model_hash[32];     // Model compatibility hash
    char reserved[32];       // Reserved for future use
} atlas_file_header_t;

// ATLAS Memory Entry Header
typedef struct {
    uint32_t key_length;     // Length of key string
    uint32_t data_size;      // Size of data payload
    uint32_t data_type;      // Type identifier for data
    uint32_t flags;          // Entry flags (compressed, encrypted, etc.)
    uint64_t timestamp;      // Entry creation/modification time
    uint32_t checksum;       // CRC32 of entry data
    uint32_t reserved;       // Reserved for alignment
} atlas_entry_header_t;

// ATLAS Memory Entry
typedef struct {
    atlas_entry_header_t header;
    char* key;               // Key string (null-terminated)
    uint8_t* data;           // Raw data payload
} atlas_memory_entry_t;

// ATLAS Persistence Context
typedef struct {
    atlas_file_header_t header;
    atlas_memory_entry_t* entries;
    size_t capacity;
    size_t count;
    bool modified;
    char filepath[512];
} atlas_persistence_t;

// Core Persistence Functions
atlas_persistence_t* atlas_persistence_create(void);
void atlas_persistence_free(atlas_persistence_t* ctx);

// File Operations
bool atlas_persistence_save(atlas_persistence_t* ctx, const char* filepath);
bool atlas_persistence_load(atlas_persistence_t* ctx, const char* filepath);
bool atlas_persistence_exists(const char* filepath);
bool atlas_persistence_validate(const char* filepath);

// Memory Operations
bool atlas_persistence_set(atlas_persistence_t* ctx, const char* key, 
                          const void* data, size_t data_size, uint32_t type);
bool atlas_persistence_get(atlas_persistence_t* ctx, const char* key,
                          void** data, size_t* data_size, uint32_t* type);
bool atlas_persistence_remove(atlas_persistence_t* ctx, const char* key);
bool atlas_persistence_has(atlas_persistence_t* ctx, const char* key);
size_t atlas_persistence_count(atlas_persistence_t* ctx);

// Iteration Functions
typedef bool (*atlas_entry_callback_t)(const char* key, const void* data, 
                                       size_t data_size, uint32_t type, void* user_data);
void atlas_persistence_foreach(atlas_persistence_t* ctx, atlas_entry_callback_t callback, void* user_data);

// Utility Functions
uint32_t atlas_crc32(const void* data, size_t size);
void atlas_generate_model_hash(const char* model_path, char hash[32]);
const char* atlas_get_auto_filename(const char* model_path, char buffer[512]);

// Error Handling
typedef enum {
    ATLAS_ERROR_NONE = 0,
    ATLAS_ERROR_INVALID_PARAM,
    ATLAS_ERROR_FILE_NOT_FOUND,
    ATLAS_ERROR_FILE_CORRUPT,
    ATLAS_ERROR_FILE_ACCESS,
    ATLAS_ERROR_MEMORY_ERROR,
    ATLAS_ERROR_KEY_NOT_FOUND,
    ATLAS_ERROR_KEY_TOO_LONG,
    ATLAS_ERROR_DATA_TOO_LARGE,
    ATLAS_ERROR_CHECKSUM_MISMATCH,
    ATLAS_ERROR_VERSION_MISMATCH,
    ATLAS_ERROR_MAGIC_MISMATCH
} atlas_error_t;

atlas_error_t atlas_get_last_error(void);
const char* atlas_error_string(atlas_error_t error);

#ifdef __cplusplus
}
#endif