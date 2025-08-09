# ATLAS-GGUF Format Specification

## Overview

ATLAS-GGUF is an extension of the standard GGUF (GPT-Generated Unified Format) that adds support for ATLAS (Advanced Tensor Learning and Attention System) metadata and tensors. This format maintains full backward compatibility with standard GGUF while providing enhanced capabilities for ATLAS-enabled models.

## Format Structure

### Standard GGUF Structure
```
[GGUF Header]
[Key-Value Pairs]
[Tensor Metadata]
[Tensor Data]
```

### ATLAS-Enhanced GGUF Structure
```
[GGUF Header]
[Standard Key-Value Pairs]
[ATLAS Key-Value Pairs]      ← New
[Standard Tensor Metadata]
[ATLAS Tensor Metadata]      ← New
[Standard Tensor Data]
[ATLAS Tensor Data]          ← New
```

## ATLAS Metadata Keys

### Core Configuration Keys

| Key | Type | Description | Required |
|-----|------|-------------|----------|
| `atlas.version` | uint32 | ATLAS format version | Yes |
| `atlas.enabled` | bool | ATLAS enhancement enabled | Yes |
| `atlas.layer_count` | uint32 | Number of model layers | No |
| `atlas.head_count` | uint32 | Number of attention heads | No |
| `atlas.batch_size` | uint32 | Default batch size | No |
| `atlas.seq_length` | uint32 | Maximum sequence length | No |
| `atlas.storage_policy` | uint32 | Storage policy (see below) | No |

### Storage Policies

| Value | Policy | Description |
|-------|--------|-------------|
| 0 | NONE | No special storage handling |
| 1 | MEMORY | Keep ATLAS data in memory |
| 2 | DISK | Store ATLAS data on disk |
| 3 | HYBRID | Mixed memory/disk storage |

### Extended Configuration Keys

| Key | Type | Description |
|-----|------|-------------|
| `atlas.config` | string | JSON configuration blob |
| `atlas.storage_version` | uint32 | Storage format version |

## ATLAS Tensor Naming Conventions

### Prefixes

- `atlas.` - General ATLAS tensors
- `atlas.attn.` - Attention-related tensors
- `atlas.ffn.` - Feed-forward network tensors
- `atlas.state.` - State/memory tensors

### Naming Pattern

```
atlas.{component}.{layer}.{parameter}
```

Examples:
- `atlas.attn.0.weight` - Layer 0 attention weights
- `atlas.attn.5.bias` - Layer 5 attention bias
- `atlas.ffn.12.up_proj` - Layer 12 FFN up projection
- `atlas.state.global.memory` - Global memory state

## File Format Validation

### Magic Number
ATLAS-GGUF files use the standard GGUF magic number: `GGUF` (0x46554747)

### Version Compatibility
- GGUF version: 3 (standard)
- ATLAS version: 1 (current)

### Validation Checklist

1. **Header Validation**
   - Magic number matches `GGUF`
   - Version is supported (≥3)
   - File structure is valid

2. **ATLAS Validation**
   - `atlas.enabled` key exists
   - `atlas.version` is supported
   - Required ATLAS keys are present

3. **Tensor Validation**
   - ATLAS tensors follow naming conventions
   - Tensor data is consistent with metadata
   - No conflicting tensor names

## API Integration

### C API Functions

```c
// Context management
atlas_gguf_context_t * atlas_gguf_init_from_file(const char * fname, struct gguf_init_params params);
atlas_gguf_context_t * atlas_gguf_init_empty(void);
void atlas_gguf_free(atlas_gguf_context_t * ctx);

// Configuration
bool atlas_gguf_load_config(const struct gguf_context * gguf_ctx, atlas_gguf_config_t * config);
bool atlas_gguf_save_config(struct gguf_context * gguf_ctx, const atlas_gguf_config_t * config);

// Tensor operations
int atlas_gguf_get_atlas_tensor_count(const struct gguf_context * gguf_ctx);
bool atlas_gguf_is_atlas_tensor(const char * name);
void atlas_gguf_make_tensor_name(char * buffer, size_t buffer_size, const char * prefix, 
                                 int layer_idx, const char * component);

// Conversion utilities
bool atlas_gguf_convert_to_atlas(const char * input_path, const char * output_path, 
                                 const atlas_gguf_config_t * config);
bool atlas_gguf_convert_from_atlas(const char * input_path, const char * output_path);
```

## Conversion Process

### Standard GGUF → ATLAS-GGUF

1. Load standard GGUF file
2. Extract model metadata (layers, heads, etc.)
3. Add ATLAS configuration keys
4. Generate ATLAS tensor placeholders
5. Write enhanced GGUF file

### ATLAS-GGUF → Standard GGUF

1. Load ATLAS-enhanced GGUF file
2. Filter out ATLAS-specific keys
3. Remove ATLAS tensors
4. Write standard GGUF file

## Usage Examples

### Loading ATLAS-GGUF File

```c
// Initialize ATLAS-GGUF context
struct gguf_init_params params = {false, NULL};
atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model_atlas.gguf", params);

if (ctx && ctx->is_atlas_enabled) {
    printf("ATLAS version: %u\n", ctx->config.version);
    printf("Layer count: %u\n", ctx->config.layer_count);
    printf("Head count: %u\n", ctx->config.head_count);
}

atlas_gguf_free(ctx);
```

### Converting to ATLAS Format

```c
// Configure ATLAS settings
atlas_gguf_config_t config = {0};
config.version = 1;
config.enabled = true;
config.layer_count = 32;
config.head_count = 32;
config.batch_size = 4;
config.seq_length = 2048;
config.storage_policy = ATLAS_STORAGE_POLICY_MEMORY;

// Convert standard GGUF to ATLAS format
bool success = atlas_gguf_convert_to_atlas("model.gguf", "model_atlas.gguf", &config);
```

### Command-Line Tools

```bash
# Show file information
./atlas-gguf-convert -i model.gguf --info

# Validate GGUF file
./atlas-gguf-convert -i model.gguf --validate

# Convert to ATLAS format
./atlas-gguf-convert -i model.gguf -o model_atlas.gguf --to-atlas \
    --layer-count 32 --head-count 32 --batch-size 4 --seq-length 2048

# Convert back to standard format
./atlas-gguf-convert -i model_atlas.gguf -o model_standard.gguf --from-atlas
```

## Error Handling

### Common Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| -1 | Invalid parameter |
| -2 | File not found |
| -3 | File corrupt |
| -4 | Version mismatch |
| -5 | Memory error |

### Error Handling Pattern

```c
atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model.gguf", params);
if (!ctx) {
    fprintf(stderr, "Failed to load ATLAS-GGUF file\n");
    return -1;
}

if (!ctx->is_atlas_enabled) {
    printf("Warning: File is not ATLAS-enhanced\n");
}
```

## Performance Considerations

### Memory Usage
- ATLAS metadata adds ~1KB per model
- ATLAS tensors size depends on model complexity
- Memory-mapped files recommended for large models

### I/O Performance
- Batch metadata operations when possible
- Use streaming for large tensor data
- Consider compression for ATLAS tensors

### Optimization Tips
1. Load ATLAS configuration once and cache
2. Use tensor name validation sparingly
3. Prefer batch operations over individual queries
4. Memory-map files for read-only access

## Backward Compatibility

### Guarantees
- Standard GGUF files load without modification
- ATLAS-enhanced files work with standard GGUF loaders (ATLAS data ignored)
- API extensions don't break existing code
- File format version remains compatible

### Migration Path
1. Standard GGUF → Load normally, optionally enhance with ATLAS
2. ATLAS-GGUF → Load with ATLAS support, or fallback to standard mode
3. Tools support both formats transparently

## Future Extensions

### Planned Features
- Compressed ATLAS tensor storage
- Incremental ATLAS data updates
- Network streaming support
- Multi-model ATLAS containers

### Version Roadmap
- v1.0: Basic ATLAS-GGUF support
- v1.1: Compression and optimization
- v2.0: Advanced ATLAS features

## Technical Specifications

### File Size Overhead
- Metadata: ~1-5KB per model
- ATLAS tensors: Variable (0-50% of model size)
- Total overhead: Typically <10% for enhanced models

### Platform Support
- Linux: Full support
- Windows: Full support
- macOS: Full support
- Mobile: Limited support (conversion tools only)

### Dependencies
- Standard C library
- GGML library
- GGUF library
- pthread (for threading support)

---

For additional information, see:
- [ATLAS Integration Guide](ATLAS_INTEGRATION.md)
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [API Reference](../include/atlas-gguf.h)