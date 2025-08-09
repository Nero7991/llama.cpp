# ATLAS-GGUF Quick Start Guide

## Introduction

This guide will help you get started with ATLAS-GGUF (GPT-Generated Unified Format with ATLAS enhancements) quickly. ATLAS-GGUF extends the standard GGUF format to support ATLAS (Advanced Tensor Learning and Attention System) features while maintaining full backward compatibility.

## Installation

### Prerequisites

- CMake 3.14 or later
- C++11 compatible compiler
- Standard GGML/llama.cpp dependencies

### Building with ATLAS-GGUF Support

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Configure build with ATLAS-GGUF enabled
cmake -B build -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build build --target llama --target atlas-gguf-convert -j4

# Verify installation
./build/atlas-gguf-convert --help
```

## Basic Usage

### 1. Inspecting GGUF Files

First, let's examine a GGUF file to understand its structure:

```bash
# Show file information
./build/atlas-gguf-convert -i model.gguf --info
```

Expected output:
```
File Information: model.gguf
==========================================
ATLAS-GGUF Information:
  Version: 1 (1.0.0)
  ATLAS Enabled: No
  Total Tensors: 291
  Total Key-Value Pairs: 24
```

### 2. Converting to ATLAS Format

Convert a standard GGUF model to ATLAS-enhanced format:

```bash
# Basic conversion
./build/atlas-gguf-convert \
    -i model.gguf \
    -o model_atlas.gguf \
    --to-atlas \
    --layer-count 32 \
    --head-count 32 \
    --batch-size 4 \
    --seq-length 2048
```

### 3. Validating Files

Validate file integrity:

```bash
# Validate original file
./build/atlas-gguf-convert -i model.gguf --validate

# Validate ATLAS-enhanced file
./build/atlas-gguf-convert -i model_atlas.gguf --validate
```

### 4. Converting Back to Standard Format

Convert ATLAS-enhanced file back to standard GGUF:

```bash
./build/atlas-gguf-convert \
    -i model_atlas.gguf \
    -o model_standard.gguf \
    --from-atlas
```

## Programming Examples

### C API Usage

#### Basic File Loading

```c
#include "atlas-gguf.h"
#include <stdio.h>

int main() {
    // Initialize GGUF parameters
    struct gguf_init_params params = {false, NULL};
    
    // Load ATLAS-GGUF file
    atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model.gguf", params);
    if (!ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }
    
    // Check if ATLAS is enabled
    if (ctx->is_atlas_enabled) {
        printf("✓ ATLAS-enhanced model loaded\n");
        printf("  Version: %u\n", ctx->config.version);
        printf("  Layers: %u\n", ctx->config.layer_count);
        printf("  Heads: %u\n", ctx->config.head_count);
    } else {
        printf("→ Standard GGUF model loaded\n");
    }
    
    // Print detailed information
    atlas_gguf_print_info(ctx);
    
    // Cleanup
    atlas_gguf_free(ctx);
    
    return 0;
}
```

#### Model Conversion

```c
#include "atlas-gguf.h"
#include <stdio.h>

int convert_to_atlas(const char* input_path, const char* output_path) {
    // Configure ATLAS settings
    atlas_gguf_config_t config = {
        .version = 1,
        .enabled = true,
        .layer_count = 32,
        .head_count = 32,
        .batch_size = 4,
        .seq_length = 2048,
        .storage_policy = ATLAS_STORAGE_POLICY_MEMORY
    };
    
    // Perform conversion
    printf("Converting %s to ATLAS format...\n", input_path);
    
    if (atlas_gguf_convert_to_atlas(input_path, output_path, &config)) {
        printf("✓ Conversion successful: %s\n", output_path);
        return 0;
    } else {
        printf("✗ Conversion failed\n");
        return 1;
    }
}

int main() {
    return convert_to_atlas("model.gguf", "model_atlas.gguf");
}
```

#### Tensor Operations

```c
#include "atlas-gguf.h"
#include <stdio.h>

void explore_tensors(atlas_gguf_context_t * ctx) {
    if (!ctx || !ctx->gguf_ctx) return;
    
    int total_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    int atlas_tensors = atlas_gguf_get_atlas_tensor_count(ctx->gguf_ctx);
    
    printf("Tensor Analysis:\n");
    printf("  Total tensors: %d\n", total_tensors);
    printf("  ATLAS tensors: %d\n", atlas_tensors);
    printf("  Standard tensors: %d\n", total_tensors - atlas_tensors);
    
    // List first few ATLAS tensors
    printf("\nATLAS Tensors:\n");
    int atlas_count = 0;
    for (int i = 0; i < total_tensors && atlas_count < 5; i++) {
        const char* name = gguf_get_tensor_name(ctx->gguf_ctx, i);
        if (atlas_gguf_is_atlas_tensor(name)) {
            printf("  [%d] %s\n", atlas_count++, name);
        }
    }
    
    if (atlas_count == 5 && atlas_tensors > 5) {
        printf("  ... and %d more ATLAS tensors\n", atlas_tensors - 5);
    }
}

int main() {
    struct gguf_init_params params = {false, NULL};
    atlas_gguf_context_t * ctx = atlas_gguf_init_from_file("model_atlas.gguf", params);
    
    if (ctx) {
        explore_tensors(ctx);
        atlas_gguf_free(ctx);
    }
    
    return 0;
}
```

## Server Integration

### Basic Server Setup

```c
#include "atlas-gguf.h"
#include "common.h"
#include "llama.h"

// Server context with ATLAS support
struct atlas_server_context {
    atlas_gguf_context_t * atlas_ctx;
    llama_context * llama_ctx;
    llama_model * llama_model;
};

int start_atlas_server(const char* model_path, int port) {
    // Initialize ATLAS context
    struct gguf_init_params params = {false, NULL};
    atlas_gguf_context_t * atlas_ctx = atlas_gguf_init_from_file(model_path, params);
    
    if (!atlas_ctx) {
        fprintf(stderr, "Failed to load ATLAS-GGUF model\n");
        return 1;
    }
    
    if (atlas_ctx->is_atlas_enabled) {
        printf("Starting ATLAS-enhanced server\n");
        atlas_gguf_print_info(atlas_ctx);
    } else {
        printf("Starting standard server with ATLAS compatibility\n");
    }
    
    // Initialize llama model (standard process)
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(model_path, model_params);
    
    if (!model) {
        fprintf(stderr, "Failed to load llama model\n");
        atlas_gguf_free(atlas_ctx);
        return 1;
    }
    
    // Create llama context
    llama_context_params ctx_params = llama_context_default_params();
    llama_context * llama_ctx = llama_new_context_with_model(model, ctx_params);
    
    if (!llama_ctx) {
        fprintf(stderr, "Failed to create llama context\n");
        llama_free_model(model);
        atlas_gguf_free(atlas_ctx);
        return 1;
    }
    
    printf("✓ Server started on port %d\n", port);
    printf("✓ Model: %s\n", model_path);
    printf("✓ ATLAS: %s\n", atlas_ctx->is_atlas_enabled ? "Enabled" : "Disabled");
    
    // Cleanup (normally server would run indefinitely)
    llama_free(llama_ctx);
    llama_free_model(model);
    atlas_gguf_free(atlas_ctx);
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [port]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int port = (argc > 2) ? atoi(argv[2]) : 8080;
    
    return start_atlas_server(model_path, port);
}
```

## Configuration Options

### ATLAS Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `version` | uint32 | 1 | ATLAS format version |
| `enabled` | bool | true | Enable ATLAS features |
| `layer_count` | uint32 | auto | Number of model layers |
| `head_count` | uint32 | auto | Number of attention heads |
| `batch_size` | uint32 | 1 | Default batch size |
| `seq_length` | uint32 | 2048 | Maximum sequence length |
| `storage_policy` | enum | MEMORY | Storage policy |

### Storage Policies

| Policy | Value | Description | Use Case |
|--------|-------|-------------|----------|
| NONE | 0 | No special handling | Testing/debugging |
| MEMORY | 1 | Keep in memory | Fast inference |
| DISK | 2 | Store on disk | Large models |
| HYBRID | 3 | Mixed approach | Balanced performance |

### Command Line Options

```bash
# Conversion tool options
./atlas-gguf-convert [OPTIONS]

Options:
  -i, --input PATH        Input GGUF file path
  -o, --output PATH       Output GGUF file path
  --to-atlas              Convert standard GGUF to ATLAS-enhanced
  --from-atlas            Convert ATLAS-enhanced GGUF to standard
  --validate              Validate GGUF file only (no conversion)
  --info                  Show file information only

ATLAS Configuration:
  --layer-count N         Number of layers (default: auto-detect)
  --head-count N          Number of attention heads (default: auto-detect)
  --batch-size N          Batch size (default: 1)
  --seq-length N          Sequence length (default: 2048)
  --storage-policy N      Storage policy: 0=none, 1=memory, 2=disk, 3=hybrid
```

## Common Workflows

### Workflow 1: Model Analysis

```bash
# 1. Inspect original model
./build/atlas-gguf-convert -i original_model.gguf --info

# 2. Validate integrity
./build/atlas-gguf-convert -i original_model.gguf --validate

# 3. Convert to ATLAS format
./build/atlas-gguf-convert -i original_model.gguf -o atlas_model.gguf --to-atlas

# 4. Compare sizes
ls -lh original_model.gguf atlas_model.gguf
```

### Workflow 2: Development Testing

```bash
# 1. Create test ATLAS model
./build/atlas-gguf-convert \
    -i test_model.gguf -o test_atlas.gguf \
    --to-atlas --layer-count 12 --head-count 12

# 2. Test with server
./build/server -m test_atlas.gguf --port 8080

# 3. Convert back for comparison
./build/atlas-gguf-convert \
    -i test_atlas.gguf -o test_restored.gguf \
    --from-atlas

# 4. Validate round-trip conversion
diff test_model.gguf test_restored.gguf
```

### Workflow 3: Production Deployment

```bash
# 1. Validate production model
./build/atlas-gguf-convert -i production_model.gguf --validate

# 2. Create optimized ATLAS version
./build/atlas-gguf-convert \
    -i production_model.gguf -o production_atlas.gguf \
    --to-atlas \
    --storage-policy 3 \
    --batch-size 8

# 3. Test ATLAS version
./build/atlas-gguf-convert -i production_atlas.gguf --info

# 4. Deploy to server
./build/server -m production_atlas.gguf --host 0.0.0.0 --port 8080
```

## Troubleshooting

### Common Issues

#### Issue: "Failed to load GGUF file"

**Solution:**
```bash
# Check file exists and is readable
ls -la model.gguf

# Validate file format
./build/atlas-gguf-convert -i model.gguf --validate

# Check file permissions
chmod 644 model.gguf
```

#### Issue: "Conversion failed"

**Solution:**
```bash
# Ensure sufficient disk space
df -h .

# Check input file validity
./build/atlas-gguf-convert -i input.gguf --info

# Try with minimal configuration
./build/atlas-gguf-convert -i input.gguf -o output.gguf --to-atlas --storage-policy 0
```

#### Issue: "Version mismatch"

**Solution:**
```bash
# Check ATLAS version compatibility
./build/atlas-gguf-convert --help | grep -i version

# Force conversion with current version
# (Advanced users only - may cause compatibility issues)
```

### Debug Mode

Enable debug output for troubleshooting:

```c
// In your code, add before calling ATLAS functions:
#define ATLAS_GGUF_DEBUG 1

// Or set environment variable:
export ATLAS_GGUF_DEBUG=1
./build/atlas-gguf-convert -i model.gguf --info
```

### Getting Help

1. **Check file compatibility:**
   ```bash
   ./build/atlas-gguf-convert -i model.gguf --validate
   ```

2. **Verify installation:**
   ```bash
   ./build/atlas-gguf-convert --help
   ```

3. **Test with minimal example:**
   ```bash
   # Create a simple test program using examples above
   ```

## Performance Tips

1. **Use appropriate storage policies:**
   - `MEMORY` for small models requiring fast access
   - `DISK` for large models with memory constraints
   - `HYBRID` for balanced performance

2. **Optimize conversion parameters:**
   - Specify exact `layer_count` and `head_count` if known
   - Use smaller `batch_size` for memory-constrained environments
   - Adjust `seq_length` based on typical usage patterns

3. **Validate files regularly:**
   - Use `--validate` before production deployment
   - Check file integrity after network transfers

## Next Steps

- Explore the [ATLAS-GGUF Format Specification](ATLAS_GGUF_FORMAT.md)
- Read the complete [API Reference](ATLAS_GGUF_API.md)  
- Check out [ATLAS Integration Guide](ATLAS_INTEGRATION.md)
- Review example implementations in the `examples/` directory

---

For more advanced usage and integration patterns, see the complete documentation in the `docs/` directory.