#include "../include/atlas-persistence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

// Simple global error storage (not thread-safe but simpler)
static atlas_error_t last_error = ATLAS_ERROR_NONE;

// CRC32 Implementation - use a simple alternative to avoid large table
static uint32_t simple_hash(const void* data, size_t size) {
    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t hash = 0x811c9dc5;  // FNV-1a initial value
    
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 0x01000193;  // FNV-1a prime
    }
    
    return hash;
}

uint32_t atlas_crc32(const void* data, size_t size) {
    return simple_hash(data, size);
}

// File existence check using stat (C++11 compatible)
static bool file_exists(const char* filepath) {
    struct stat st;
    return stat(filepath, &st) == 0;
}

// Error handling functions
atlas_error_t atlas_get_last_error(void) {
    return last_error;
}

const char* atlas_error_string(atlas_error_t error) {
    switch (error) {
        case ATLAS_ERROR_NONE: return "No error";
        case ATLAS_ERROR_INVALID_PARAM: return "Invalid parameter";
        case ATLAS_ERROR_FILE_NOT_FOUND: return "File not found";
        case ATLAS_ERROR_FILE_CORRUPT: return "File corrupted";
        case ATLAS_ERROR_FILE_ACCESS: return "File access error";
        case ATLAS_ERROR_MEMORY_ERROR: return "Memory allocation error";
        case ATLAS_ERROR_KEY_NOT_FOUND: return "Key not found";
        case ATLAS_ERROR_KEY_TOO_LONG: return "Key too long";
        case ATLAS_ERROR_DATA_TOO_LARGE: return "Data too large";
        case ATLAS_ERROR_CHECKSUM_MISMATCH: return "Checksum mismatch";
        case ATLAS_ERROR_VERSION_MISMATCH: return "Version mismatch";
        case ATLAS_ERROR_MAGIC_MISMATCH: return "Magic number mismatch";
        default: return "Unknown error";
    }
}

// Create new persistence context
atlas_persistence_t* atlas_persistence_create(void) {
    atlas_persistence_t* ctx = (atlas_persistence_t*)calloc(1, sizeof(atlas_persistence_t));
    if (!ctx) {
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return nullptr;
    }
    
    // Initialize header
    ctx->header.magic = ATLAS_MAGIC;
    ctx->header.version = ATLAS_VERSION;
    ctx->header.created_time = time(nullptr);
    ctx->header.modified_time = ctx->header.created_time;
    ctx->header.entry_count = 0;
    ctx->header.total_size = 0;
    
    // Initialize entries array
    ctx->capacity = 64;  // Start with small capacity
    ctx->count = 0;
    ctx->entries = (atlas_memory_entry_t*)calloc(ctx->capacity, sizeof(atlas_memory_entry_t));
    if (!ctx->entries) {
        free(ctx);
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return nullptr;
    }
    
    ctx->modified = false;
    memset(ctx->filepath, 0, sizeof(ctx->filepath));
    
    last_error = ATLAS_ERROR_NONE;
    return ctx;
}

// Free persistence context
void atlas_persistence_free(atlas_persistence_t* ctx) {
    if (!ctx) return;
    
    // Free all entries
    if (ctx->entries) {
        for (size_t i = 0; i < ctx->count; i++) {
            free(ctx->entries[i].key);
            free(ctx->entries[i].data);
        }
        free(ctx->entries);
    }
    
    free(ctx);
}

// Find entry by key
static atlas_memory_entry_t* find_entry(atlas_persistence_t* ctx, const char* key) {
    if (!ctx || !key) return nullptr;
    
    for (size_t i = 0; i < ctx->count; i++) {
        if (ctx->entries[i].key && strcmp(ctx->entries[i].key, key) == 0) {
            return &ctx->entries[i];
        }
    }
    return nullptr;
}

// Resize entries array if needed
static bool ensure_capacity(atlas_persistence_t* ctx, size_t needed_capacity) {
    if (ctx->capacity >= needed_capacity) return true;
    
    size_t new_capacity = ctx->capacity;
    while (new_capacity < needed_capacity) {
        new_capacity *= 2;
    }
    
    if (new_capacity > ATLAS_MAX_ENTRIES) {
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return false;
    }
    
    atlas_memory_entry_t* new_entries = (atlas_memory_entry_t*)realloc(
        ctx->entries, new_capacity * sizeof(atlas_memory_entry_t));
    if (!new_entries) {
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return false;
    }
    
    // Zero out new entries
    memset(&new_entries[ctx->capacity], 0, 
           (new_capacity - ctx->capacity) * sizeof(atlas_memory_entry_t));
    
    ctx->entries = new_entries;
    ctx->capacity = new_capacity;
    return true;
}

// Set memory entry
bool atlas_persistence_set(atlas_persistence_t* ctx, const char* key, 
                          const void* data, size_t data_size, uint32_t type) {
    if (!ctx || !key || !data || data_size == 0) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    if (strlen(key) > ATLAS_MAX_KEY_LENGTH) {
        last_error = ATLAS_ERROR_KEY_TOO_LONG;
        return false;
    }
    
    if (data_size > ATLAS_MAX_DATA_SIZE) {
        last_error = ATLAS_ERROR_DATA_TOO_LARGE;
        return false;
    }
    
    // Find existing entry or create new one
    atlas_memory_entry_t* entry = find_entry(ctx, key);
    if (!entry) {
        // Need new entry
        if (!ensure_capacity(ctx, ctx->count + 1)) {
            return false;
        }
        entry = &ctx->entries[ctx->count++];
        
        // Allocate key
        entry->key = (char*)malloc(strlen(key) + 1);
        if (!entry->key) {
            ctx->count--;  // Revert count increment
            last_error = ATLAS_ERROR_MEMORY_ERROR;
            return false;
        }
        strcpy(entry->key, key);
    } else {
        // Update existing entry - free old data
        free(entry->data);
    }
    
    // Allocate and copy data
    entry->data = (uint8_t*)malloc(data_size);
    if (!entry->data) {
        if (entry->key && find_entry(ctx, key) != entry) {
            // This was a new entry, clean up
            free(entry->key);
            ctx->count--;
        }
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return false;
    }
    
    memcpy(entry->data, data, data_size);
    
    // Update header
    entry->header.key_length = (uint32_t)strlen(key);
    entry->header.data_size = (uint32_t)data_size;
    entry->header.data_type = type;
    entry->header.timestamp = time(nullptr);
    entry->header.checksum = atlas_crc32(data, data_size);
    entry->header.flags = 0;
    
    // Update context
    ctx->modified = true;
    ctx->header.modified_time = time(nullptr);
    ctx->header.entry_count = (uint32_t)ctx->count;
    
    last_error = ATLAS_ERROR_NONE;
    return true;
}

// Get memory entry
bool atlas_persistence_get(atlas_persistence_t* ctx, const char* key,
                          void** data, size_t* data_size, uint32_t* type) {
    if (!ctx || !key || !data || !data_size) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    atlas_memory_entry_t* entry = find_entry(ctx, key);
    if (!entry) {
        last_error = ATLAS_ERROR_KEY_NOT_FOUND;
        return false;
    }
    
    // Verify checksum
    uint32_t computed_checksum = atlas_crc32(entry->data, entry->header.data_size);
    if (computed_checksum != entry->header.checksum) {
        last_error = ATLAS_ERROR_CHECKSUM_MISMATCH;
        return false;
    }
    
    // Allocate and copy data for caller
    *data = malloc(entry->header.data_size);
    if (!*data) {
        last_error = ATLAS_ERROR_MEMORY_ERROR;
        return false;
    }
    
    memcpy(*data, entry->data, entry->header.data_size);
    *data_size = entry->header.data_size;
    if (type) *type = entry->header.data_type;
    
    last_error = ATLAS_ERROR_NONE;
    return true;
}

// Check if key exists
bool atlas_persistence_has(atlas_persistence_t* ctx, const char* key) {
    if (!ctx || !key) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    return find_entry(ctx, key) != nullptr;
}

// Remove entry
bool atlas_persistence_remove(atlas_persistence_t* ctx, const char* key) {
    if (!ctx || !key) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    for (size_t i = 0; i < ctx->count; i++) {
        if (ctx->entries[i].key && strcmp(ctx->entries[i].key, key) == 0) {
            // Free entry data
            free(ctx->entries[i].key);
            free(ctx->entries[i].data);
            
            // Shift remaining entries
            if (i < ctx->count - 1) {
                memmove(&ctx->entries[i], &ctx->entries[i + 1], 
                       (ctx->count - i - 1) * sizeof(atlas_memory_entry_t));
            }
            
            ctx->count--;
            ctx->modified = true;
            ctx->header.entry_count = (uint32_t)ctx->count;
            ctx->header.modified_time = time(nullptr);
            
            last_error = ATLAS_ERROR_NONE;
            return true;
        }
    }
    
    last_error = ATLAS_ERROR_KEY_NOT_FOUND;
    return false;
}

// Get entry count
size_t atlas_persistence_count(atlas_persistence_t* ctx) {
    if (!ctx) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return 0;
    }
    return ctx->count;
}

// Save to file
bool atlas_persistence_save(atlas_persistence_t* ctx, const char* filepath) {
    if (!ctx || !filepath) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    FILE* file = fopen(filepath, "wb");
    if (!file) {
        last_error = ATLAS_ERROR_FILE_ACCESS;
        return false;
    }
    
    // Calculate total data size and update header
    ctx->header.total_size = 0;
    for (size_t i = 0; i < ctx->count; i++) {
        ctx->header.total_size += sizeof(atlas_entry_header_t) + 
                                 ctx->entries[i].header.key_length + 1 + 
                                 ctx->entries[i].header.data_size;
    }
    
    // Calculate header checksum (excluding checksum field itself)
    ctx->header.checksum = 0;
    uint8_t* header_bytes = (uint8_t*)&ctx->header;
    ctx->header.checksum = atlas_crc32(header_bytes + sizeof(uint32_t) * 3, 
                                      sizeof(atlas_file_header_t) - sizeof(uint32_t) * 3);
    
    // Write header
    if (fwrite(&ctx->header, sizeof(atlas_file_header_t), 1, file) != 1) {
        fclose(file);
        last_error = ATLAS_ERROR_FILE_ACCESS;
        return false;
    }
    
    // Write entries
    for (size_t i = 0; i < ctx->count; i++) {
        atlas_memory_entry_t* entry = &ctx->entries[i];
        
        // Write entry header
        if (fwrite(&entry->header, sizeof(atlas_entry_header_t), 1, file) != 1) {
            fclose(file);
            last_error = ATLAS_ERROR_FILE_ACCESS;
            return false;
        }
        
        // Write key (with null terminator)
        if (fwrite(entry->key, entry->header.key_length + 1, 1, file) != 1) {
            fclose(file);
            last_error = ATLAS_ERROR_FILE_ACCESS;
            return false;
        }
        
        // Write data
        if (fwrite(entry->data, entry->header.data_size, 1, file) != 1) {
            fclose(file);
            last_error = ATLAS_ERROR_FILE_ACCESS;
            return false;
        }
    }
    
    fclose(file);
    
    // Update context state
    ctx->modified = false;
    strncpy(ctx->filepath, filepath, sizeof(ctx->filepath) - 1);
    ctx->filepath[sizeof(ctx->filepath) - 1] = '\0';
    
    last_error = ATLAS_ERROR_NONE;
    return true;
}

// Load from file
bool atlas_persistence_load(atlas_persistence_t* ctx, const char* filepath) {
    if (!ctx || !filepath) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    if (!file_exists(filepath)) {
        last_error = ATLAS_ERROR_FILE_NOT_FOUND;
        return false;
    }
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        last_error = ATLAS_ERROR_FILE_ACCESS;
        return false;
    }
    
    // Read header
    atlas_file_header_t header;
    if (fread(&header, sizeof(atlas_file_header_t), 1, file) != 1) {
        fclose(file);
        last_error = ATLAS_ERROR_FILE_CORRUPT;
        return false;
    }
    
    // Validate header
    if (header.magic != ATLAS_MAGIC) {
        fclose(file);
        last_error = ATLAS_ERROR_MAGIC_MISMATCH;
        return false;
    }
    
    if (header.version != ATLAS_VERSION) {
        fclose(file);
        last_error = ATLAS_ERROR_VERSION_MISMATCH;
        return false;
    }
    
    // Clear existing entries
    if (ctx->entries) {
        for (size_t i = 0; i < ctx->count; i++) {
            free(ctx->entries[i].key);
            free(ctx->entries[i].data);
        }
        free(ctx->entries);
    }
    
    // Allocate space for entries
    if (!ensure_capacity(ctx, header.entry_count)) {
        fclose(file);
        return false;
    }
    
    ctx->count = 0;
    ctx->header = header;
    
    // Read entries
    for (uint32_t i = 0; i < header.entry_count; i++) {
        atlas_memory_entry_t* entry = &ctx->entries[i];
        
        // Read entry header
        if (fread(&entry->header, sizeof(atlas_entry_header_t), 1, file) != 1) {
            fclose(file);
            last_error = ATLAS_ERROR_FILE_CORRUPT;
            return false;
        }
        
        // Allocate and read key
        entry->key = (char*)malloc(entry->header.key_length + 1);
        if (!entry->key) {
            fclose(file);
            last_error = ATLAS_ERROR_MEMORY_ERROR;
            return false;
        }
        
        if (fread(entry->key, entry->header.key_length + 1, 1, file) != 1) {
            fclose(file);
            free(entry->key);
            last_error = ATLAS_ERROR_FILE_CORRUPT;
            return false;
        }
        
        // Allocate and read data
        entry->data = (uint8_t*)malloc(entry->header.data_size);
        if (!entry->data) {
            fclose(file);
            free(entry->key);
            last_error = ATLAS_ERROR_MEMORY_ERROR;
            return false;
        }
        
        if (fread(entry->data, entry->header.data_size, 1, file) != 1) {
            fclose(file);
            free(entry->key);
            free(entry->data);
            last_error = ATLAS_ERROR_FILE_CORRUPT;
            return false;
        }
        
        // Verify checksum
        uint32_t computed_checksum = atlas_crc32(entry->data, entry->header.data_size);
        if (computed_checksum != entry->header.checksum) {
            fclose(file);
            free(entry->key);
            free(entry->data);
            last_error = ATLAS_ERROR_CHECKSUM_MISMATCH;
            return false;
        }
        
        ctx->count++;
    }
    
    fclose(file);
    
    // Update context state
    ctx->modified = false;
    strncpy(ctx->filepath, filepath, sizeof(ctx->filepath) - 1);
    ctx->filepath[sizeof(ctx->filepath) - 1] = '\0';
    
    last_error = ATLAS_ERROR_NONE;
    return true;
}

// Check if file exists
bool atlas_persistence_exists(const char* filepath) {
    if (!filepath) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    return file_exists(filepath);
}

// Validate file format
bool atlas_persistence_validate(const char* filepath) {
    if (!filepath) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return false;
    }
    
    if (!file_exists(filepath)) {
        last_error = ATLAS_ERROR_FILE_NOT_FOUND;
        return false;
    }
    
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        last_error = ATLAS_ERROR_FILE_ACCESS;
        return false;
    }
    
    atlas_file_header_t header;
    if (fread(&header, sizeof(atlas_file_header_t), 1, file) != 1) {
        fclose(file);
        last_error = ATLAS_ERROR_FILE_CORRUPT;
        return false;
    }
    
    fclose(file);
    
    if (header.magic != ATLAS_MAGIC) {
        last_error = ATLAS_ERROR_MAGIC_MISMATCH;
        return false;
    }
    
    if (header.version != ATLAS_VERSION) {
        last_error = ATLAS_ERROR_VERSION_MISMATCH;
        return false;
    }
    
    last_error = ATLAS_ERROR_NONE;
    return true;
}

// Generate automatic filename
const char* atlas_get_auto_filename(const char* model_path, char buffer[512]) {
    if (!model_path || !buffer) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return nullptr;
    }
    
    // Extract base name from model path
    const char* base_name = strrchr(model_path, '/');
    if (!base_name) {
        base_name = strrchr(model_path, '\\');  // Windows path
    }
    if (!base_name) {
        base_name = model_path;
    } else {
        base_name++;  // Skip path separator
    }
    
    // Remove .gguf extension if present
    char clean_name[256];
    strncpy(clean_name, base_name, sizeof(clean_name) - 1);
    clean_name[sizeof(clean_name) - 1] = '\0';
    
    char* ext = strstr(clean_name, ".gguf");
    if (ext) *ext = '\0';
    
    // Generate filename
    snprintf(buffer, 512, "%s_memory.atlas", clean_name);
    
    last_error = ATLAS_ERROR_NONE;
    return buffer;
}

// Iterate over entries
void atlas_persistence_foreach(atlas_persistence_t* ctx, atlas_entry_callback_t callback, void* user_data) {
    if (!ctx || !callback) {
        last_error = ATLAS_ERROR_INVALID_PARAM;
        return;
    }
    
    for (size_t i = 0; i < ctx->count; i++) {
        atlas_memory_entry_t* entry = &ctx->entries[i];
        if (!callback(entry->key, entry->data, entry->header.data_size, 
                     entry->header.data_type, user_data)) {
            break;  // Callback requested termination
        }
    }
    
    last_error = ATLAS_ERROR_NONE;
}