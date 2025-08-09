# ATLAS Runtime API Reference

## Overview

This document provides detailed API documentation for the ATLAS (Advanced Tensor Learning and Attention System) Runtime initialization system. The ATLAS Runtime enables dynamic optimization of GGUF models without requiring file conversion.

## Table of Contents

- [Data Types](#data-types)
- [Core Functions](#core-functions)
- [Architecture Detection](#architecture-detection)
- [Memory Management](#memory-management)
- [Module Management](#module-management)
- [Statistics & Monitoring](#statistics--monitoring)
- [Error Handling](#error-handling)
- [Constants](#constants)

## Data Types

### `atlas_runtime_ctx`
```cpp
typedef struct atlas_runtime_ctx atlas_runtime_ctx;
```
Opaque handle representing an ATLAS runtime instance. Contains all state and resources needed for ATLAS operations.

### `atlas_runtime_params`
```cpp
typedef struct atlas_runtime_params {
    bool enable_atlas;           // Enable ATLAS optimizations (default: true)
    bool auto_detect_arch;       // Auto-detect model architecture (default: true)
    float sparsity_threshold;    // Sparsity threshold 0.0-1.0 (default: 0.1)
    int max_modules;            // Maximum optimization modules (default: 16)
    size_t memory_pool_size;    // Memory pool size in bytes (default: 256MB)
    const char * arch_override; // Override architecture detection (default: NULL)
    bool enable_lazy_loading;   // Enable lazy module loading (default: true)
    float optimization_level;   // Optimization level 0.0-2.0 (default: 1.0)
} atlas_runtime_params;
```
Configuration parameters for ATLAS runtime initialization.

**Field Details:**
- `enable_atlas`: Master switch for ATLAS functionality
- `auto_detect_arch`: When true, automatically detects model architecture
- `sparsity_threshold`: Controls trade-off between speed and accuracy (lower = more accurate)
- `max_modules`: Limits number of concurrent optimization modules
- `memory_pool_size`: Size of ATLAS memory pool (minimum 1MB)
- `arch_override`: Force specific architecture ("llama", "mistral", "phi", "gemma")
- `enable_lazy_loading`: Load optimization modules on-demand
- `optimization_level`: Optimization aggressiveness (0.0 = conservative, 2.0 = maximum)

### `atlas_runtime_result`
```cpp
typedef enum {
    ATLAS_RUNTIME_SUCCESS = 0,
    ATLAS_RUNTIME_ERROR_INVALID_PARAMS,
    ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION,
    ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED,
    ATLAS_RUNTIME_ERROR_INIT_FAILED,
    ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND,
    ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED,
    ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL
} atlas_runtime_result;
```
Result codes returned by ATLAS runtime functions.

### `atlas_arch_type`
```cpp
typedef enum {
    ATLAS_ARCH_AUTO = 0,        // Auto-detect architecture
    ATLAS_ARCH_LLAMA,          // Llama/Llama2 architecture
    ATLAS_ARCH_MISTRAL,        // Mistral architecture
    ATLAS_ARCH_PHI,            // Microsoft Phi architecture
    ATLAS_ARCH_GEMMA,          // Google Gemma architecture
    ATLAS_ARCH_UNKNOWN         // Unknown/unsupported architecture
} atlas_arch_type;
```
Supported model architectures.

### `atlas_arch_config`
```cpp
typedef struct atlas_arch_config {
    atlas_arch_type type;           // Architecture type
    const char * name;              // Human-readable name
    float sparsity_default;         // Default sparsity threshold
    int layers_optimal;             // Optimal layer count for this architecture
    size_t memory_requirements;     // Estimated memory requirements
    bool supports_dynamic_loading;  // Supports dynamic module loading
} atlas_arch_config;
```
Architecture-specific configuration parameters.

### `atlas_module_info`
```cpp
typedef struct atlas_module_info {
    const char * name;          // Module name
    atlas_arch_type arch_type;  // Target architecture
    size_t memory_usage;        // Memory usage in bytes
    bool is_active;            // Whether module is currently active
    float performance_gain;     // Measured performance improvement
} atlas_module_info;
```
Information about loaded optimization modules.

### `atlas_runtime_stats`
```cpp
typedef struct atlas_runtime_stats {
    size_t total_memory_used;      // Total memory used by ATLAS
    size_t modules_loaded;         // Number of loaded modules
    float average_speedup;         // Average performance speedup
    uint64_t operations_processed; // Total operations processed
} atlas_runtime_stats;
```
Runtime performance statistics.

## Core Functions

### `atlas_runtime_init`
```cpp
atlas_runtime_result atlas_runtime_init(
    atlas_runtime_ctx ** ctx,
    const struct llama_model * model,
    const atlas_runtime_params * params
);
```

Initializes ATLAS runtime with specified parameters.

**Parameters:**
- `ctx`: [out] Pointer to receive runtime context handle
- `model`: [in] llama.cpp model to optimize (can be NULL)
- `params`: [in] Configuration parameters (NULL for defaults)

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Initialization successful
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: Invalid parameters
- `ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION`: Memory allocation failed
- `ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED`: Model architecture not supported
- `ATLAS_RUNTIME_ERROR_INIT_FAILED`: General initialization failure

**Example:**
```cpp
atlas_runtime_params params = atlas_runtime_default_params();
params.memory_pool_size = 128 * 1024 * 1024;  // 128MB

atlas_runtime_ctx * ctx = nullptr;
atlas_runtime_result result = atlas_runtime_init(&ctx, model, &params);

if (result != ATLAS_RUNTIME_SUCCESS) {
    fprintf(stderr, "ATLAS init failed: %s\n", 
            atlas_runtime_error_string(result));
    return false;
}
```

**Thread Safety:** Thread-safe. Multiple threads can call simultaneously.

### `atlas_runtime_cleanup`
```cpp
atlas_runtime_result atlas_runtime_cleanup(atlas_runtime_ctx * ctx);
```

Cleans up ATLAS runtime and releases all resources.

**Parameters:**
- `ctx`: [in] Runtime context to cleanup

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Cleanup successful
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: Invalid context

**Example:**
```cpp
atlas_runtime_cleanup(ctx);
ctx = nullptr;  // Important: set to NULL after cleanup
```

**Thread Safety:** Not thread-safe. Ensure no other threads are using the context.

### `atlas_runtime_enable_for_context`
```cpp
atlas_runtime_result atlas_runtime_enable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
);
```

Enables ATLAS optimizations for a specific llama.cpp context.

**Parameters:**
- `runtime_ctx`: [in] ATLAS runtime context
- `llama_ctx`: [in] llama.cpp context to optimize

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Successfully enabled
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: Invalid parameters
- `ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL`: Context incompatible with ATLAS

**Thread Safety:** Thread-safe with different contexts.

### `atlas_runtime_disable_for_context`
```cpp
atlas_runtime_result atlas_runtime_disable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
);
```

Disables ATLAS optimizations for a specific llama.cpp context.

**Thread Safety:** Thread-safe with different contexts.

### `atlas_runtime_default_params`
```cpp
atlas_runtime_params atlas_runtime_default_params(void);
```

Returns default ATLAS runtime parameters.

**Returns:** `atlas_runtime_params` structure with default values.

**Default Values:**
```cpp
{
    .enable_atlas = true,
    .auto_detect_arch = true,
    .sparsity_threshold = 0.1f,
    .max_modules = 16,
    .memory_pool_size = 256 * 1024 * 1024,  // 256MB
    .arch_override = NULL,
    .enable_lazy_loading = true,
    .optimization_level = 1.0f
}
```

### `atlas_runtime_validate_params`
```cpp
bool atlas_runtime_validate_params(const atlas_runtime_params * params);
```

Validates ATLAS runtime parameters.

**Parameters:**
- `params`: [in] Parameters to validate

**Returns:**
- `true`: Parameters are valid
- `false`: One or more parameters are invalid

**Validation Rules:**
- `sparsity_threshold`: Must be between 0.0 and 1.0
- `max_modules`: Must be between 1 and 1024
- `memory_pool_size`: Must be at least 1MB
- `optimization_level`: Must be between 0.0 and 2.0

## Architecture Detection

### `atlas_runtime_detect_architecture`
```cpp
atlas_arch_type atlas_runtime_detect_architecture(const struct llama_model * model);
```

Automatically detects the architecture of a llama.cpp model.

**Parameters:**
- `model`: [in] llama.cpp model to analyze

**Returns:** Detected architecture type or `ATLAS_ARCH_UNKNOWN`

**Detection Method:**
1. Examines model metadata for architecture string
2. Analyzes model structure (layer count, dimensions, etc.)
3. Uses heuristics to classify unknown models

### `atlas_runtime_get_arch_config`
```cpp
const atlas_arch_config * atlas_runtime_get_arch_config(atlas_arch_type arch);
```

Retrieves configuration parameters for a specific architecture.

**Parameters:**
- `arch`: [in] Architecture type

**Returns:** Pointer to configuration structure (never NULL)

**Example:**
```cpp
const atlas_arch_config * config = atlas_runtime_get_arch_config(ATLAS_ARCH_LLAMA);
printf("Architecture: %s\n", config->name);
printf("Default sparsity: %.2f\n", config->sparsity_default);
printf("Memory requirements: %zu MB\n", config->memory_requirements / (1024 * 1024));
```

### `atlas_runtime_arch_name`
```cpp
const char * atlas_runtime_arch_name(atlas_arch_type arch);
```

Returns human-readable name for architecture type.

**Parameters:**
- `arch`: [in] Architecture type

**Returns:** String name (never NULL)

## Memory Management

### `atlas_runtime_get_pool_usage`
```cpp
size_t atlas_runtime_get_pool_usage(atlas_runtime_ctx * ctx);
```

Returns current memory pool usage in bytes.

**Thread Safety:** Thread-safe.

### `atlas_runtime_get_pool_capacity`
```cpp
size_t atlas_runtime_get_pool_capacity(atlas_runtime_ctx * ctx);
```

Returns total memory pool capacity in bytes.

**Thread Safety:** Thread-safe.

### `atlas_runtime_resize_pool`
```cpp
atlas_runtime_result atlas_runtime_resize_pool(
    atlas_runtime_ctx * ctx,
    size_t new_size
);
```

Resizes the ATLAS memory pool.

**Parameters:**
- `ctx`: [in] Runtime context
- `new_size`: [in] New pool size in bytes (minimum 1MB)

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Pool resized successfully
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: Invalid size
- `ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION`: Resize failed

**Notes:**
- Resizing may temporarily use additional memory during transition
- Active allocations are preserved during resize
- Shrinking below current usage will fail

**Thread Safety:** Thread-safe, but may block other memory operations.

## Module Management

### `atlas_runtime_load_module`
```cpp
atlas_runtime_result atlas_runtime_load_module(
    atlas_runtime_ctx * ctx,
    const char * module_name,
    const atlas_arch_config * config
);
```

Loads an optimization module for specific architecture.

**Parameters:**
- `ctx`: [in] Runtime context
- `module_name`: [in] Module name to load
- `config`: [in] Architecture configuration (NULL for auto)

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Module loaded successfully
- `ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND`: Module not found
- `ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION`: Insufficient memory

**Available Modules:**
- `"attention_optimization"`: Optimizes attention computation
- `"kv_cache_optimization"`: Optimizes KV cache usage
- `"quantization_aware"`: Quantization-aware optimizations
- `"sparse_inference"`: Sparse inference optimizations

### `atlas_runtime_unload_module`
```cpp
atlas_runtime_result atlas_runtime_unload_module(
    atlas_runtime_ctx * ctx,
    const char * module_name
);
```

Unloads a previously loaded optimization module.

**Thread Safety:** Thread-safe.

### `atlas_runtime_get_active_modules`
```cpp
int atlas_runtime_get_active_modules(
    atlas_runtime_ctx * ctx,
    atlas_module_info * modules,
    int max_modules
);
```

Retrieves information about active optimization modules.

**Parameters:**
- `ctx`: [in] Runtime context
- `modules`: [out] Array to receive module information
- `max_modules`: [in] Maximum number of modules to return

**Returns:** Number of modules returned (may be less than requested)

**Example:**
```cpp
atlas_module_info modules[32];
int count = atlas_runtime_get_active_modules(ctx, modules, 32);

for (int i = 0; i < count; ++i) {
    printf("Module: %s, Memory: %zu KB, Speedup: %.2fx\n",
           modules[i].name,
           modules[i].memory_usage / 1024,
           modules[i].performance_gain);
}
```

## Statistics & Monitoring

### `atlas_runtime_get_stats`
```cpp
atlas_runtime_result atlas_runtime_get_stats(
    atlas_runtime_ctx * ctx,
    atlas_runtime_stats * stats
);
```

Retrieves runtime performance statistics.

**Parameters:**
- `ctx`: [in] Runtime context
- `stats`: [out] Statistics structure to fill

**Returns:**
- `ATLAS_RUNTIME_SUCCESS`: Statistics retrieved
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: Invalid parameters

**Example:**
```cpp
atlas_runtime_stats stats;
if (atlas_runtime_get_stats(ctx, &stats) == ATLAS_RUNTIME_SUCCESS) {
    printf("Memory usage: %zu MB\n", stats.total_memory_used / (1024 * 1024));
    printf("Modules loaded: %zu\n", stats.modules_loaded);
    printf("Average speedup: %.2fx\n", stats.average_speedup);
    printf("Operations: %lu\n", stats.operations_processed);
}
```

**Thread Safety:** Thread-safe.

### `atlas_runtime_reset_stats`
```cpp
atlas_runtime_result atlas_runtime_reset_stats(atlas_runtime_ctx * ctx);
```

Resets runtime statistics counters.

**Thread Safety:** Thread-safe.

## Error Handling

### `atlas_runtime_error_string`
```cpp
const char * atlas_runtime_error_string(atlas_runtime_result error);
```

Converts error code to human-readable string.

**Parameters:**
- `error`: [in] Error code

**Returns:** Error description string (never NULL)

**Error Messages:**
- `ATLAS_RUNTIME_SUCCESS`: "Success"
- `ATLAS_RUNTIME_ERROR_INVALID_PARAMS`: "Invalid parameters"
- `ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION`: "Memory allocation failed"
- `ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED`: "Architecture not supported"
- `ATLAS_RUNTIME_ERROR_INIT_FAILED`: "Initialization failed"
- `ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND`: "Module not found"
- `ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED`: "Memory pool exhausted"
- `ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL`: "Incompatible model"

### `atlas_runtime_get_last_error`
```cpp
atlas_runtime_result atlas_runtime_get_last_error(void);
```

Returns the last error code for the current thread.

**Returns:** Last error code (thread-local)

**Notes:**
- Error state is thread-local
- Reset to `ATLAS_RUNTIME_SUCCESS` after successful operations
- Useful for debugging when error code is not directly available

## Constants

### Memory Limits
```cpp
#define ATLAS_MIN_POOL_SIZE     (1024 * 1024)      // 1MB minimum pool
#define ATLAS_MAX_POOL_SIZE     (8ULL * 1024 * 1024 * 1024)  // 8GB maximum
#define ATLAS_DEFAULT_POOL_SIZE (256 * 1024 * 1024) // 256MB default
```

### Module Limits
```cpp
#define ATLAS_MIN_MODULES       1       // Minimum modules
#define ATLAS_MAX_MODULES       1024    // Maximum modules  
#define ATLAS_DEFAULT_MODULES   16      // Default modules
```

### Optimization Levels
```cpp
#define ATLAS_OPT_LEVEL_MIN     0.0f    // Conservative optimizations
#define ATLAS_OPT_LEVEL_DEFAULT 1.0f    // Balanced optimizations
#define ATLAS_OPT_LEVEL_MAX     2.0f    // Aggressive optimizations
```

### Sparsity Thresholds
```cpp
#define ATLAS_SPARSITY_MIN      0.0f    // No sparsity
#define ATLAS_SPARSITY_DEFAULT  0.1f    // 10% sparsity
#define ATLAS_SPARSITY_MAX      1.0f    // Maximum sparsity
```

## Usage Patterns

### Basic Initialization Pattern
```cpp
// 1. Initialize ATLAS
atlas_runtime_params params = atlas_runtime_default_params();
atlas_runtime_ctx * ctx = nullptr;
atlas_runtime_result result = atlas_runtime_init(&ctx, model, &params);

if (result != ATLAS_RUNTIME_SUCCESS) {
    // Handle error
    return false;
}

// 2. Enable for contexts
atlas_runtime_enable_for_context(ctx, llama_ctx);

// 3. Use normally
// ... inference code ...

// 4. Cleanup
atlas_runtime_cleanup(ctx);
```

### Error Handling Pattern
```cpp
atlas_runtime_result result = atlas_runtime_init(&ctx, model, &params);

switch (result) {
case ATLAS_RUNTIME_SUCCESS:
    break;
case ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED:
    printf("Warning: Architecture not supported, continuing without ATLAS\n");
    break;
case ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION:
    printf("Error: Insufficient memory for ATLAS\n");
    return false;
default:
    printf("Error: %s\n", atlas_runtime_error_string(result));
    return false;
}
```

### Resource Management Pattern
```cpp
// RAII-style wrapper for C++
class AtlasRuntime {
    atlas_runtime_ctx * ctx = nullptr;
    
public:
    AtlasRuntime(const atlas_runtime_params & params, llama_model * model) {
        atlas_runtime_init(&ctx, model, &params);
    }
    
    ~AtlasRuntime() {
        if (ctx) {
            atlas_runtime_cleanup(ctx);
        }
    }
    
    bool is_valid() const { return ctx != nullptr; }
    atlas_runtime_ctx * get() const { return ctx; }
    
    // Non-copyable
    AtlasRuntime(const AtlasRuntime &) = delete;
    AtlasRuntime & operator=(const AtlasRuntime &) = delete;
};
```

## Performance Considerations

### Memory Pool Sizing Guidelines

| Use Case | Recommended Pool Size |
|----------|----------------------|
| Single user chat | 64-128 MB |
| Batch processing | 256-512 MB |
| Server (multi-user) | 512 MB - 2 GB |
| Development/testing | 32-64 MB |

### Optimization Level Guidelines

| Level | Description | Use Case |
|-------|-------------|----------|
| 0.0-0.5 | Conservative | Production, accuracy-critical |
| 0.5-1.5 | Balanced | General use, good speed/accuracy |
| 1.5-2.0 | Aggressive | Speed-critical, can tolerate minor accuracy loss |

### Thread Safety Summary

| Function | Thread Safety | Notes |
|----------|---------------|-------|
| `atlas_runtime_init` | Thread-safe | Multiple simultaneous calls OK |
| `atlas_runtime_cleanup` | Not thread-safe | No concurrent access to same context |
| `atlas_runtime_enable_for_context` | Thread-safe | With different contexts |
| Memory pool functions | Thread-safe | Internal synchronization |
| Statistics functions | Thread-safe | Read-only or atomic updates |
| Module management | Thread-safe | Internal synchronization |