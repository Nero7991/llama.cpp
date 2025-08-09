# ATLAS-GGUF API Reference

## Overview

The ATLAS-GGUF API provides C-compatible functions for working with ATLAS-enhanced GGUF files. This API extends the standard GGUF functionality while maintaining full backward compatibility.

## Core Data Structures

### atlas_gguf_context_t

Main context structure for ATLAS-GGUF operations.

```c
typedef struct {
    struct gguf_context * gguf_ctx;
    atlas_gguf_config_t config;
    bool is_atlas_enabled;
} atlas_gguf_context_t;
```

### atlas_gguf_config_t

Configuration structure for ATLAS settings.

```c
typedef struct {
    uint32_t version;
    bool enabled;
    uint32_t layer_count;
    uint32_t head_count;
    uint32_t batch_size;
    uint32_t seq_length;
    atlas_storage_policy_t storage_policy;
} atlas_gguf_config_t;
```

### atlas_storage_policy_t

Storage policy enumeration.

```c
typedef enum {
    ATLAS_STORAGE_POLICY_NONE = 0,
    ATLAS_STORAGE_POLICY_MEMORY,
    ATLAS_STORAGE_POLICY_DISK,
    ATLAS_STORAGE_POLICY_HYBRID,
} atlas_storage_policy_t;
```

## Context Management

### atlas_gguf_init_from_file

Initialize ATLAS-GGUF context from file.

```c
atlas_gguf_context_t * atlas_gguf_init_from_file(
    const char * fname, 
    struct gguf_init_params params
);
```

**Parameters:**
- `fname`: Path to GGUF file
- `params`: GGUF initialization parameters

**Returns:**
- Pointer to initialized context, or `NULL` on failure

**Example:**
```c
struct gguf_init_params params = {false, NULL};
atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model.gguf", params);
if (ctx) {
    // Use context
    atlas_gguf_free(ctx);
}
```

### atlas_gguf_init_empty

Initialize empty ATLAS-GGUF context.

```c
atlas_gguf_context_t * atlas_gguf_init_empty(void);
```

**Returns:**
- Pointer to initialized context, or `NULL` on failure

**Example:**
```c
atlas_gguf_context_t * ctx = atlas_gguf_init_empty();
if (ctx) {
    // Configure and use context
    atlas_gguf_free(ctx);
}
```

### atlas_gguf_free

Free ATLAS-GGUF context and associated resources.

```c
void atlas_gguf_free(atlas_gguf_context_t * ctx);
```

**Parameters:**
- `ctx`: Context to free (can be `NULL`)

**Example:**
```c
atlas_gguf_free(ctx);
ctx = NULL; // Good practice
```

## Configuration Operations

### atlas_gguf_has_atlas_data

Check if GGUF file contains ATLAS metadata.

```c
bool atlas_gguf_has_atlas_data(const struct gguf_context * gguf_ctx);
```

**Parameters:**
- `gguf_ctx`: GGUF context to check

**Returns:**
- `true` if ATLAS data is present, `false` otherwise

### atlas_gguf_load_config

Load ATLAS configuration from GGUF context.

```c
bool atlas_gguf_load_config(
    const struct gguf_context * gguf_ctx, 
    atlas_gguf_config_t * config
);
```

**Parameters:**
- `gguf_ctx`: Source GGUF context
- `config`: Output configuration structure

**Returns:**
- `true` on success, `false` on failure

**Example:**
```c
atlas_gguf_config_t config;
if (atlas_gguf_load_config(ctx->gguf_ctx, &config)) {
    printf("ATLAS version: %u\n", config.version);
    printf("Enabled: %s\n", config.enabled ? "Yes" : "No");
}
```

### atlas_gguf_save_config

Save ATLAS configuration to GGUF context.

```c
bool atlas_gguf_save_config(
    struct gguf_context * gguf_ctx, 
    const atlas_gguf_config_t * config
);
```

**Parameters:**
- `gguf_ctx`: Target GGUF context
- `config`: Configuration to save

**Returns:**
- `true` on success, `false` on failure

**Example:**
```c
atlas_gguf_config_t config = {
    .version = 1,
    .enabled = true,
    .layer_count = 32,
    .head_count = 32,
    .batch_size = 4,
    .seq_length = 2048,
    .storage_policy = ATLAS_STORAGE_POLICY_MEMORY
};

if (atlas_gguf_save_config(ctx->gguf_ctx, &config)) {
    printf("Configuration saved successfully\n");
}
```

## Tensor Operations

### atlas_gguf_get_atlas_tensor_count

Get the number of ATLAS tensors in the file.

```c
int atlas_gguf_get_atlas_tensor_count(const struct gguf_context * gguf_ctx);
```

**Parameters:**
- `gguf_ctx`: GGUF context to query

**Returns:**
- Number of ATLAS tensors, or 0 if none found

### atlas_gguf_is_atlas_tensor

Check if a tensor name follows ATLAS naming conventions.

```c
bool atlas_gguf_is_atlas_tensor(const char * name);
```

**Parameters:**
- `name`: Tensor name to check

**Returns:**
- `true` if name is ATLAS tensor, `false` otherwise

**Example:**
```c
const char* names[] = {
    "atlas.attn.0.weight",     // true
    "atlas.ffn.5.bias",        // true
    "blk.0.attn_k.weight",     // false
    "output.weight"            // false
};

for (int i = 0; i < 4; i++) {
    printf("%s is ATLAS: %s\n", names[i], 
           atlas_gguf_is_atlas_tensor(names[i]) ? "Yes" : "No");
}
```

### atlas_gguf_make_tensor_name

Generate ATLAS tensor names following conventions.

```c
void atlas_gguf_make_tensor_name(
    char * buffer, 
    size_t buffer_size, 
    const char * prefix, 
    int layer_idx, 
    const char * component
);
```

**Parameters:**
- `buffer`: Output buffer for tensor name
- `buffer_size`: Size of output buffer
- `prefix`: Tensor prefix (e.g., `ATLAS_TENSOR_ATTENTION_PREFIX`)
- `layer_idx`: Layer index (-1 for no layer)
- `component`: Component name (e.g., "weight", "bias")

**Example:**
```c
char name[256];

atlas_gguf_make_tensor_name(name, sizeof(name), 
    ATLAS_TENSOR_ATTENTION_PREFIX, 5, "weight");
// Result: "atlas.attn.5.weight"

atlas_gguf_make_tensor_name(name, sizeof(name), 
    ATLAS_TENSOR_FFN_PREFIX, 12, "bias");
// Result: "atlas.ffn.12.bias"
```

## Conversion Functions

### atlas_gguf_convert_to_atlas

Convert standard GGUF to ATLAS-enhanced format.

```c
bool atlas_gguf_convert_to_atlas(
    const char * input_path, 
    const char * output_path, 
    const atlas_gguf_config_t * config
);
```

**Parameters:**
- `input_path`: Path to input GGUF file
- `output_path`: Path for output ATLAS-GGUF file
- `config`: ATLAS configuration for conversion

**Returns:**
- `true` on success, `false` on failure

**Example:**
```c
atlas_gguf_config_t config = {
    .version = 1,
    .enabled = true,
    .layer_count = 32,
    .head_count = 32,
    .storage_policy = ATLAS_STORAGE_POLICY_MEMORY
};

if (atlas_gguf_convert_to_atlas("model.gguf", "model_atlas.gguf", &config)) {
    printf("Conversion to ATLAS format successful\n");
}
```

### atlas_gguf_convert_from_atlas

Convert ATLAS-enhanced GGUF to standard format.

```c
bool atlas_gguf_convert_from_atlas(
    const char * input_path, 
    const char * output_path
);
```

**Parameters:**
- `input_path`: Path to input ATLAS-GGUF file
- `output_path`: Path for output standard GGUF file

**Returns:**
- `true` on success, `false` on failure

**Example:**
```c
if (atlas_gguf_convert_from_atlas("model_atlas.gguf", "model_standard.gguf")) {
    printf("Conversion to standard format successful\n");
}
```

## Validation Functions

### atlas_gguf_validate

Validate ATLAS-GGUF file integrity.

```c
bool atlas_gguf_validate(
    const char * path, 
    char * error_msg, 
    size_t error_msg_size
);
```

**Parameters:**
- `path`: Path to file to validate
- `error_msg`: Buffer for error message (can be `NULL`)
- `error_msg_size`: Size of error message buffer

**Returns:**
- `true` if valid, `false` otherwise

**Example:**
```c
char error[256];
if (atlas_gguf_validate("model.gguf", error, sizeof(error))) {
    printf("File validation passed\n");
} else {
    printf("Validation failed: %s\n", error);
}
```

## Utility Functions

### atlas_gguf_get_version_string

Get ATLAS-GGUF version string.

```c
const char * atlas_gguf_get_version_string(void);
```

**Returns:**
- Version string (e.g., "1.0.0")

### atlas_gguf_print_info

Print ATLAS-GGUF context information.

```c
void atlas_gguf_print_info(const atlas_gguf_context_t * ctx);
```

**Parameters:**
- `ctx`: Context to print information for

**Example:**
```c
atlas_gguf_print_info(ctx);
// Output:
// ATLAS-GGUF Information:
//   Version: 1 (1.0.0)
//   ATLAS Enabled: Yes
//   Configuration:
//     Layer Count: 32
//     Head Count: 32
//     ...
```

## Constants and Macros

### ATLAS Metadata Keys

```c
#define ATLAS_GGUF_KEY_VERSION          "atlas.version"
#define ATLAS_GGUF_KEY_ENABLED          "atlas.enabled"
#define ATLAS_GGUF_KEY_CONFIG           "atlas.config"
#define ATLAS_GGUF_KEY_LAYER_COUNT      "atlas.layer_count"
#define ATLAS_GGUF_KEY_HEAD_COUNT       "atlas.head_count"
#define ATLAS_GGUF_KEY_BATCH_SIZE       "atlas.batch_size"
#define ATLAS_GGUF_KEY_SEQ_LENGTH       "atlas.seq_length"
#define ATLAS_GGUF_KEY_STORAGE_POLICY   "atlas.storage_policy"
```

### ATLAS Tensor Prefixes

```c
#define ATLAS_TENSOR_PREFIX             "atlas."
#define ATLAS_TENSOR_ATTENTION_PREFIX   "atlas.attn."
#define ATLAS_TENSOR_FFN_PREFIX         "atlas.ffn."
#define ATLAS_TENSOR_STATE_PREFIX       "atlas.state."
```

## Error Handling

### Best Practices

1. **Always check return values:**
```c
atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model.gguf", params);
if (!ctx) {
    fprintf(stderr, "Failed to initialize ATLAS-GGUF context\n");
    return -1;
}
```

2. **Use validation functions:**
```c
char error[256];
if (!atlas_gguf_validate("model.gguf", error, sizeof(error))) {
    fprintf(stderr, "Validation error: %s\n", error);
    return -1;
}
```

3. **Free resources properly:**
```c
// Always free contexts when done
atlas_gguf_free(ctx);
ctx = NULL;
```

## Thread Safety

- **Context objects**: Not thread-safe, use separate contexts per thread
- **Read operations**: Safe across multiple threads with same context
- **Write operations**: Require external synchronization
- **Conversion functions**: Thread-safe (stateless)

## Performance Tips

1. **Reuse contexts** when processing multiple operations
2. **Validate files once** and cache results
3. **Use memory-mapped files** for read-only access
4. **Batch tensor operations** when possible

## Integration Examples

### Basic File Loading

```c
#include "atlas-gguf.h"

int main() {
    struct gguf_init_params params = {false, NULL};
    atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model.gguf", params);
    
    if (!ctx) {
        printf("Failed to load file\n");
        return 1;
    }
    
    if (ctx->is_atlas_enabled) {
        printf("ATLAS-enhanced model loaded\n");
        atlas_gguf_print_info(ctx);
    } else {
        printf("Standard GGUF model loaded\n");
    }
    
    atlas_gguf_free(ctx);
    return 0;
}
```

### Model Conversion

```c
#include "atlas-gguf.h"

int convert_model(const char* input, const char* output) {
    // Validate input file
    char error[256];
    if (!atlas_gguf_validate(input, error, sizeof(error))) {
        fprintf(stderr, "Input validation failed: %s\n", error);
        return -1;
    }
    
    // Configure ATLAS settings
    atlas_gguf_config_t config = {0};
    config.version = 1;
    config.enabled = true;
    config.layer_count = 32;
    config.head_count = 32;
    config.storage_policy = ATLAS_STORAGE_POLICY_MEMORY;
    
    // Perform conversion
    if (!atlas_gguf_convert_to_atlas(input, output, &config)) {
        fprintf(stderr, "Conversion failed\n");
        return -1;
    }
    
    printf("Conversion successful: %s -> %s\n", input, output);
    return 0;
}
```

### Tensor Enumeration

```c
void list_atlas_tensors(atlas_gguf_context_t * ctx) {
    if (!ctx || !ctx->gguf_ctx) return;
    
    int total_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    int atlas_count = 0;
    
    printf("ATLAS Tensors:\n");
    for (int i = 0; i < total_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx->gguf_ctx, i);
        if (atlas_gguf_is_atlas_tensor(name)) {
            printf("  [%d] %s\n", atlas_count++, name);
        }
    }
    
    printf("Found %d ATLAS tensors out of %d total tensors\n", 
           atlas_count, total_tensors);
}
```

---

For more information, see:
- [ATLAS-GGUF Format Specification](ATLAS_GGUF_FORMAT.md)
- [ATLAS Integration Guide](ATLAS_INTEGRATION.md)
- [Header file](../include/atlas-gguf.h)