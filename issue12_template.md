## Summary

Implement ATLAS memory persistence system for saving and loading test-time memorization state across sessions. Enable automatic memory persistence with configurable file naming and graceful recovery mechanisms.

## Background

ATLAS memory modules learn valuable context-specific information during test-time. This knowledge should persist across server restarts, allowing models to maintain long-term memory of documents, conversations, and domain-specific adaptations.

## Key Requirements

### Automatic Memory Persistence
- Save ATLAS memory automatically on graceful shutdown: `<model_name>_memory.atlas`
- Load existing memory file automatically on startup when ATLAS enabled
- Allow custom memory file specification via command-line or API
- Fallback to fresh memory initialization if no file exists or loading fails

### Memory File Format
- Binary format optimized for fast save/load operations
- Include metadata: model compatibility, creation timestamp, memory statistics
- Compression support to minimize file size
- Integrity checking with checksums

## Implementation Requirements

### 1. Memory File Format Specification

#### ATLAS Memory File Structure (.atlas extension)
```c
struct atlas_memory_header {
    char magic[8];              // "ATLASVER"
    uint32_t version;           // File format version
    uint32_t checksum;          // CRC32 of data section
    uint64_t created_timestamp; // Unix timestamp
    uint64_t model_hash;        // Model compatibility hash
    
    // Memory configuration
    uint32_t n_layers;          // Number of layers
    uint32_t memory_dim;        // Memory module dimension
    uint32_t window_size;       // Sliding window size
    uint32_t polynomial_degree; // Feature mapping degree
    
    // Statistics
    uint64_t total_tokens_seen; // Total tokens processed
    float avg_loss;             // Average optimization loss
    uint32_t save_count;        // Number of times saved
    
    uint32_t data_size;         // Size of data section in bytes
    uint32_t compressed;        // 1 if compressed, 0 if raw
};

struct atlas_memory_data {
    // Deep memory module weights per layer
    float* memory_weights;      // [n_layers][memory_params]
    
    // Muon optimizer state
    float* momentum_buffers;    // [n_layers][memory_params]
    float* hessian_approx;      // [n_layers][memory_params][memory_params]
    
    // Sliding window context (recent)
    float* window_keys;         // [window_size][key_dim]
    float* window_values;       // [window_size][value_dim]
    uint32_t window_head;       // Current window position
    
    // Feature mapping parameters
    float* polynomial_coeffs;   // [n_layers][polynomial_degree+1]
    
    // Optimization statistics
    float* layer_losses;        // [n_layers] per-layer loss history
    uint32_t* update_counts;    // [n_layers] update counters
};
```

### 2. Core Memory Persistence Functions

#### Save Operations
```c
// Save ATLAS memory to file
int atlas_save_memory(
    const struct atlas_context* ctx,
    const char* filename,
    bool compress,
    const struct atlas_save_options* options
);

// Auto-save with default naming
int atlas_auto_save_memory(
    const struct atlas_context* ctx,
    const char* model_name
);

// Serialize memory to buffer (for testing/debugging)
size_t atlas_serialize_memory(
    const struct atlas_context* ctx,
    void* buffer,
    size_t buffer_size
);

struct atlas_save_options {
    bool include_optimizer_state;  // Save Muon momentum/Hessian
    bool include_sliding_window;   // Save recent context window
    bool compress_data;            // Use compression (zlib/lz4)
    int compression_level;         // 1-9 for compression quality
    bool include_statistics;       // Save performance statistics
};
```

#### Load Operations
```c
// Load ATLAS memory from file
int atlas_load_memory(
    struct atlas_context* ctx,
    const char* filename,
    const struct atlas_load_options* options
);

// Auto-load with default naming and fallback
int atlas_auto_load_memory(
    struct atlas_context* ctx,
    const char* model_name,
    bool fallback_to_fresh
);

// Validate memory file compatibility
int atlas_validate_memory_file(
    const char* filename,
    const struct atlas_context* ctx,
    struct atlas_memory_info* info
);

struct atlas_load_options {
    bool reset_optimizer_state;    // Reset Muon state to initial
    bool clear_sliding_window;     // Clear context window
    bool strict_compatibility;     // Fail on any compatibility issues
    float learning_rate_override;  // Override saved learning rate
    bool merge_with_existing;      // Merge instead of replace
};

struct atlas_memory_info {
    char model_name[256];
    uint64_t created_timestamp;
    uint64_t total_tokens_seen;
    uint32_t n_layers;
    uint32_t memory_dim;
    float avg_loss;
    bool is_compatible;
};
```

### 3. Command-Line Integration

#### Server Command-Line Parameters
```bash
# Basic ATLAS with auto-persistence
./llama-server -m model.gguf --atlas --atlas-memory-persist

# Custom memory file
./llama-server -m model.gguf --atlas --atlas-memory-file custom_session.atlas

# Advanced persistence options
./llama-server -m model.gguf \
    --atlas \
    --atlas-memory-persist \
    --atlas-memory-dir ./atlas_memories/ \
    --atlas-auto-save-interval 300 \
    --atlas-memory-compress \
    --atlas-memory-backup-count 3

# Disable specific memory components
./llama-server -m model.gguf \
    --atlas \
    --atlas-memory-file session.atlas \
    --atlas-no-load-optimizer \
    --atlas-no-load-window
```

#### CLI Tool Integration
```bash
# Standard inference with persistence
./llama-cli -m model.gguf \
    --atlas \
    --atlas-memory-file document_analysis.atlas \
    -f long_document.txt

# Inspect memory file
./llama-atlas-inspect --file session.atlas --verbose

# Convert/migrate memory files
./llama-atlas-convert --input old_format.atlas --output new_format.atlas

# Merge multiple memory files
./llama-atlas-merge --output combined.atlas file1.atlas file2.atlas
```

### 4. Graceful Shutdown Handling

#### Signal Handling for Memory Persistence
```c
// Enhanced shutdown handler
void atlas_shutdown_handler(int signal) {
    printf("Received shutdown signal %d, saving ATLAS memory...\n", signal);
    
    // Save all active ATLAS contexts
    for (int i = 0; i < num_active_contexts; i++) {
        struct atlas_context* ctx = &active_contexts[i];
        
        if (ctx->memory_file_path[0] != '\0') {
            // Use specified memory file
            atlas_save_memory(ctx, ctx->memory_file_path, true, &default_save_options);
        } else {
            // Auto-generate filename
            char auto_filename[512];
            snprintf(auto_filename, sizeof(auto_filename), 
                     "%s_memory.atlas", ctx->model_name);
            atlas_save_memory(ctx, auto_filename, true, &default_save_options);
        }
        
        printf("Saved ATLAS memory for context %d to %s\n", i, 
               ctx->memory_file_path[0] ? ctx->memory_file_path : auto_filename);
    }
    
    printf("ATLAS memory persistence complete.\n");
    exit(0);
}

// Register signal handlers
void atlas_register_shutdown_handlers() {
    signal(SIGTERM, atlas_shutdown_handler);
    signal(SIGINT, atlas_shutdown_handler);
    signal(SIGUSR1, atlas_manual_save_handler);  // Manual save trigger
}
```

### 5. Backup and Recovery Mechanisms

#### Automatic Backup System
```c
// Backup configuration
struct atlas_backup_config {
    bool enabled;
    int max_backup_count;       // Keep N most recent backups
    int auto_backup_interval;   // Backup every N seconds
    char backup_directory[512]; // Directory for backup files
    bool compress_backups;      // Compress backup files
};

// Create backup with timestamp
int atlas_create_backup(
    const struct atlas_context* ctx,
    const struct atlas_backup_config* config
) {
    time_t now = time(NULL);
    char backup_filename[512];
    snprintf(backup_filename, sizeof(backup_filename),
             "%s/%s_memory_%ld.atlas.bak",
             config->backup_directory,
             ctx->model_name,
             now);
    
    return atlas_save_memory(ctx, backup_filename, 
                           config->compress_backups, &backup_save_options);
}

// Cleanup old backups
void atlas_cleanup_old_backups(
    const char* model_name,
    const struct atlas_backup_config* config
);

// Recovery from backup
int atlas_recover_from_backup(
    struct atlas_context* ctx,
    const char* model_name,
    int backup_index  // 0 = most recent, 1 = second most recent, etc.
);
```

### 6. File System Integration

#### Memory File Management
```c
// Memory file utilities
bool atlas_memory_file_exists(const char* filename);
size_t atlas_get_memory_file_size(const char* filename);
int atlas_verify_memory_file_integrity(const char* filename);

// Directory management
int atlas_create_memory_directory(const char* path);
int atlas_list_memory_files(const char* directory, char*** filenames, int* count);
int atlas_clean_memory_directory(const char* directory, int max_files);

// File locking for concurrent access
int atlas_lock_memory_file(const char* filename);
void atlas_unlock_memory_file(const char* filename);

// Atomic file operations
int atlas_atomic_save_memory(
    const struct atlas_context* ctx,
    const char* filename,
    const char* temp_suffix
);
```

## Testing Requirements

### Memory Persistence Tests
- [ ] **Save/load cycle**: Memory state identical after save/load
- [ ] **Auto-persistence**: Automatic save on shutdown, load on startup
- [ ] **File format compatibility**: Forward/backward compatibility across versions
- [ ] **Corruption recovery**: Graceful handling of corrupted memory files
- [ ] **Large memory files**: Performance with multi-GB memory states

### Integration Tests
- [ ] **Server integration**: llama-server with memory persistence
- [ ] **CLI integration**: llama-cli with custom memory files
- [ ] **Concurrent access**: Multiple processes accessing memory files safely
- [ ] **Cross-platform**: Memory files work across Linux/Windows/macOS

### Edge Case Tests
- [ ] **Disk full**: Graceful handling when disk space exhausted
- [ ] **Permission errors**: Appropriate errors for file permission issues
- [ ] **Model mismatch**: Proper validation when loading incompatible memory
- [ ] **Incomplete files**: Recovery from partially written memory files

## Implementation Files

### Core Persistence
- `src/atlas/atlas-persistence.h` - Memory persistence API declarations
- `src/atlas/atlas-persistence.cpp` - Core save/load implementation
- `src/atlas/atlas-file-format.h` - File format definitions
- `src/atlas/atlas-compression.cpp` - Compression/decompression utilities

### File Management
- `src/atlas/atlas-file-utils.cpp` - File system utilities
- `src/atlas/atlas-backup.cpp` - Backup and recovery system
- `src/atlas/atlas-validation.cpp` - File validation and integrity checking

### Integration
- `examples/server/atlas-server-persistence.cpp` - Server persistence integration
- `examples/atlas-cli-persistence.cpp` - CLI tool persistence integration
- `tools/atlas-inspect.cpp` - Memory file inspection tool

### Test Files
- `tests/atlas/test-persistence.cpp` - Core persistence functionality
- `tests/atlas/test-file-format.cpp` - File format validation
- `tests/atlas/test-backup-recovery.cpp` - Backup/recovery testing

## Success Criteria

### Functional Requirements
- [ ] Memory state persists correctly across all shutdown/restart cycles
- [ ] Auto-naming convention works: `<model_name>_memory.atlas`
- [ ] Custom memory file specification works via CLI and API
- [ ] Backup system maintains configurable number of historical saves
- [ ] File corruption detection and recovery mechanisms function properly

### Performance Requirements
- [ ] Memory save operations complete in <5 seconds for typical sizes
- [ ] Memory load operations complete in <2 seconds
- [ ] Compressed memory files are <50% of uncompressed size
- [ ] No performance impact during normal operation (only save/load)

### Quality Requirements
- [ ] 100% memory state fidelity after save/load cycles
- [ ] Zero data loss during graceful shutdown
- [ ] Robust error handling for all file system edge cases
- [ ] Cross-platform compatibility for memory file format

## File Naming Conventions

### Default Auto-naming
```
<model_name>_memory.atlas           # Current memory state
<model_name>_memory.atlas.bak       # Previous backup
<model_name>_memory_<timestamp>.atlas.bak  # Historical backups
```

### Custom Naming Examples
```
# Document-specific memory
document_analysis_memory.atlas
legal_brief_analysis.atlas

# Session-specific memory  
chat_session_12345.atlas
qa_session_technical.atlas

# Domain-specific memory
medical_domain.atlas
finance_domain.atlas
```

## Dependencies
- Issues #3-11: All ATLAS components and API integration
- File compression library (zlib or lz4)
- File system utilities
- Signal handling support

## Estimated Effort
**2 weeks** for experienced systems programmer

## Usage Examples

### Basic Persistence
```bash
# Server with auto-persistence
./llama-server -m llama-7b-chat.gguf --atlas --atlas-memory-persist

# On shutdown, saves to: llama-7b-chat_memory.atlas
# On restart, loads from: llama-7b-chat_memory.atlas (if exists)
```

### Custom Memory Files
```bash
# Specify custom memory file
./llama-server -m model.gguf --atlas --atlas-memory-file my_session.atlas

# CLI with document-specific memory
./llama-cli -m model.gguf --atlas --atlas-memory-file legal_docs.atlas -f contract.txt
```

### Advanced Configuration
```bash
# Server with backup and compression
./llama-server -m model.gguf \
    --atlas \
    --atlas-memory-persist \
    --atlas-memory-dir ./memories/ \
    --atlas-memory-backup-count 5 \
    --atlas-auto-save-interval 300 \
    --atlas-memory-compress
```

## References
- Binary file format best practices
- Memory-mapped file techniques
- Signal handling in Unix systems
- Data compression algorithms
