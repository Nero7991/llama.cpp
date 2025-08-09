# ATLAS Runtime Initialization Guide

## Overview

The ATLAS (Advanced Tensor Learning and Attention System) Runtime provides dynamic optimization capabilities for any GGUF model without requiring file conversion. This system enables runtime initialization of ATLAS modules that can accelerate inference, optimize memory usage, and provide architecture-specific enhancements.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Integration Guide](#integration-guide)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Quick Start

### Basic Usage

```cpp
#include "atlas-runtime.h"
#include "llama.h"

int main() {
    // Initialize ATLAS runtime
    atlas_runtime_params params = atlas_runtime_default_params();
    params.memory_pool_size = 64 * 1024 * 1024; // 64MB
    params.auto_detect_arch = true;
    
    atlas_runtime_ctx * runtime_ctx = nullptr;
    atlas_runtime_result result = atlas_runtime_init(&runtime_ctx, model, &params);
    
    if (result != ATLAS_RUNTIME_SUCCESS) {
        fprintf(stderr, "ATLAS initialization failed: %s\n", 
                atlas_runtime_error_string(result));
        return 1;
    }
    
    // Load your GGUF model normally
    llama_backend_init();
    llama_model * model = llama_load_model_from_file("model.gguf", model_params);
    
    // Enable ATLAS for context
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    atlas_runtime_enable_for_context(runtime_ctx, ctx);
    
    // Use model normally - ATLAS optimizations apply automatically
    // ... inference code ...
    
    // Cleanup
    atlas_runtime_cleanup(runtime_ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}
```

### Command Line Usage

```bash
# Enable ATLAS runtime with default settings
./llama-cli --model model.gguf --atlas-enable

# Configure ATLAS memory pool
./llama-cli --model model.gguf --atlas-enable --atlas-memory-pool 128M

# Override architecture detection
./llama-cli --model model.gguf --atlas-enable --atlas-arch llama

# Enable specific optimizations
./llama-cli --model model.gguf --atlas-enable --atlas-sparsity 0.15
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ATLAS Runtime System                     │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │  Runtime Core   │ │ Architecture    │ │  Memory Pool    │ │
│ │                 │ │   Detection     │ │   Manager       │ │
│ │ • Initialization│ │ • Model Analysis│ │ • Pool Alloc    │ │
│ │ • Module Mgmt   │ │ • Arch Configs  │ │ • Thread Safety │ │
│ │ • Context Mgmt  │ │ • Capabilities  │ │ • Statistics    │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    llama.cpp Integration                    │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│ │   Model Layer   │ │  Context Layer  │ │ Backend Layer   │ │
│ │                 │ │                 │ │                 │ │
│ │ • Model Loading │ │ • Inference     │ │ • GPU/CPU Mgmt  │ │
│ │ • Architecture  │ │ • Token Proc    │ │ • Memory Mgmt   │ │
│ │ • Metadata      │ │ • Batch Proc    │ │ • Quantization  │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Supported Architectures

| Architecture | Support Level | Key Features |
|--------------|---------------|--------------|
| **Llama**    | Full         | Attention optimization, KV cache, RoPE |
| **Mistral**  | Full         | GQA support, sliding window attention |
| **Phi**      | Full         | Compact architecture, efficient inference |
| **Gemma**    | Full         | RMSNorm, SwiGLU activation |
| **Mixtral**  | Partial      | MoE routing, sparse expert selection |
| **Other**    | Basic        | Generic optimizations only |

### Memory Management

ATLAS uses a custom memory pool system optimized for AI workloads:

- **Thread-safe allocation** - Concurrent access from multiple threads
- **Alignment optimization** - SIMD-friendly memory layout (32-byte aligned)
- **Fragmentation reduction** - Smart allocation strategies
- **Statistics tracking** - Memory usage monitoring and profiling

## API Reference

### Core Functions

#### `atlas_runtime_init`
```cpp
atlas_runtime_result atlas_runtime_init(
    atlas_runtime_ctx ** ctx,
    const struct llama_model * model,
    const atlas_runtime_params * params
);
```
Initializes ATLAS runtime with the given parameters.

**Parameters:**
- `ctx`: Output pointer for runtime context
- `model`: llama.cpp model to optimize (can be NULL for generic init)
- `params`: Configuration parameters (NULL for defaults)

**Returns:** `ATLAS_RUNTIME_SUCCESS` on success, error code otherwise.

#### `atlas_runtime_cleanup`
```cpp
atlas_runtime_result atlas_runtime_cleanup(atlas_runtime_ctx * ctx);
```
Cleans up ATLAS runtime and frees all resources.

#### `atlas_runtime_enable_for_context`
```cpp
atlas_runtime_result atlas_runtime_enable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
);
```
Enables ATLAS optimizations for a specific llama.cpp context.

### Configuration

#### `atlas_runtime_params`
```cpp
typedef struct atlas_runtime_params {
    bool enable_atlas;           // Enable ATLAS optimizations
    bool auto_detect_arch;       // Automatically detect model architecture
    float sparsity_threshold;    // Sparsity threshold for optimizations
    int max_modules;            // Maximum number of optimization modules
    size_t memory_pool_size;    // Memory pool size in bytes
    const char * arch_override; // Override detected architecture
    bool enable_lazy_loading;   // Enable lazy module loading
    float optimization_level;   // Optimization level (0.0-2.0)
} atlas_runtime_params;
```

#### Default Parameters
```cpp
atlas_runtime_params atlas_runtime_default_params(void);
```
Returns default configuration suitable for most use cases:
- Memory pool: 256MB
- Auto-detect architecture: Enabled
- Sparsity threshold: 0.1
- Max modules: 16
- Optimization level: 1.0

### Architecture Detection

#### `atlas_runtime_detect_architecture`
```cpp
atlas_arch_type atlas_runtime_detect_architecture(const struct llama_model * model);
```
Analyzes model and returns detected architecture type.

#### `atlas_runtime_get_arch_config`
```cpp
const atlas_arch_config * atlas_runtime_get_arch_config(atlas_arch_type arch);
```
Returns configuration parameters for specific architecture.

### Memory Pool Operations

#### `atlas_runtime_get_pool_usage`
```cpp
size_t atlas_runtime_get_pool_usage(atlas_runtime_ctx * ctx);
```
Returns current memory pool usage in bytes.

#### `atlas_runtime_get_pool_capacity`
```cpp
size_t atlas_runtime_get_pool_capacity(atlas_runtime_ctx * ctx);
```
Returns total memory pool capacity in bytes.

### Error Handling

#### Error Codes
```cpp
typedef enum {
    ATLAS_RUNTIME_SUCCESS = 0,           // No error
    ATLAS_RUNTIME_ERROR_INVALID_PARAMS,  // Invalid parameters
    ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION, // Memory allocation failed
    ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED,  // Architecture not supported
    ATLAS_RUNTIME_ERROR_INIT_FAILED,       // Initialization failed
    ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND,  // Module not found
    ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED,    // Memory pool exhausted
    ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL // Model incompatible with ATLAS
} atlas_runtime_result;
```

#### Error Handling
```cpp
const char * atlas_runtime_error_string(atlas_runtime_result error);
atlas_runtime_result atlas_runtime_get_last_error(void);
```

## Integration Guide

### llama.cpp Integration

ATLAS integrates seamlessly with existing llama.cpp workflows:

#### Model Loading
```cpp
// Standard model loading
llama_model_params model_params = llama_model_default_params();
llama_model * model = llama_load_model_from_file("model.gguf", model_params);

// Initialize ATLAS after model loading
atlas_runtime_params atlas_params = atlas_runtime_default_params();
atlas_runtime_ctx * runtime_ctx = nullptr;
atlas_runtime_init(&runtime_ctx, model, &atlas_params);
```

#### Context Creation
```cpp
llama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 4096;

llama_context * ctx = llama_new_context_with_model(model, ctx_params);

// Enable ATLAS for this context
atlas_runtime_enable_for_context(runtime_ctx, ctx);
```

#### Inference
```cpp
// Inference works exactly the same - optimizations are transparent
std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
int result = llama_eval(ctx, tokens.data(), tokens.size(), 0, 1);

// Get ATLAS performance metrics
atlas_runtime_stats stats;
atlas_runtime_get_stats(runtime_ctx, &stats);
printf("ATLAS speedup: %.2fx\n", stats.average_speedup);
```

### CLI Integration

Add ATLAS support to existing CLI applications:

```cpp
// Add ATLAS arguments to argument parser
bool enable_atlas = false;
size_t atlas_memory_pool = 256 * 1024 * 1024;
std::string atlas_arch_override;

// In argument parsing loop:
if (arg == "--atlas-enable") {
    enable_atlas = true;
} else if (arg == "--atlas-memory-pool") {
    atlas_memory_pool = std::stoull(argv[++i]);
} else if (arg == "--atlas-arch") {
    atlas_arch_override = argv[++i];
}

// Initialize ATLAS if requested
atlas_runtime_ctx * runtime_ctx = nullptr;
if (enable_atlas) {
    atlas_runtime_params params = atlas_runtime_default_params();
    params.memory_pool_size = atlas_memory_pool;
    if (!atlas_arch_override.empty()) {
        params.arch_override = atlas_arch_override.c_str();
        params.auto_detect_arch = false;
    }
    
    atlas_runtime_init(&runtime_ctx, model, &params);
    atlas_runtime_enable_for_context(runtime_ctx, ctx);
}
```

### Server Integration

For llama-server and similar applications:

```cpp
struct server_params {
    // ... existing parameters ...
    bool atlas_enable = false;
    size_t atlas_memory_pool_size = 256 * 1024 * 1024;
    std::string atlas_arch_override;
    float atlas_sparsity_threshold = 0.1f;
};

class llama_server {
private:
    atlas_runtime_ctx * atlas_runtime = nullptr;
    
public:
    bool initialize_atlas() {
        if (!params.atlas_enable) return true;
        
        atlas_runtime_params atlas_params = atlas_runtime_default_params();
        atlas_params.memory_pool_size = params.atlas_memory_pool_size;
        atlas_params.sparsity_threshold = params.atlas_sparsity_threshold;
        
        if (!params.atlas_arch_override.empty()) {
            atlas_params.arch_override = params.atlas_arch_override.c_str();
            atlas_params.auto_detect_arch = false;
        }
        
        atlas_runtime_result result = atlas_runtime_init(&atlas_runtime, model, &atlas_params);
        return result == ATLAS_RUNTIME_SUCCESS;
    }
    
    void enable_atlas_for_slot(llama_context * ctx) {
        if (atlas_runtime) {
            atlas_runtime_enable_for_context(atlas_runtime, ctx);
        }
    }
};
```

## Performance Tuning

### Memory Pool Sizing

Choose memory pool size based on model size and usage:

```cpp
atlas_runtime_params params = atlas_runtime_default_params();

// For small models (< 7B parameters)
params.memory_pool_size = 64 * 1024 * 1024;   // 64MB

// For medium models (7B-13B parameters)  
params.memory_pool_size = 128 * 1024 * 1024;  // 128MB

// For large models (> 13B parameters)
params.memory_pool_size = 256 * 1024 * 1024;  // 256MB

// For server applications with multiple concurrent requests
params.memory_pool_size = 512 * 1024 * 1024;  // 512MB
```

### Sparsity Threshold Tuning

Adjust sparsity threshold based on accuracy requirements:

```cpp
// Conservative (higher accuracy, less speedup)
params.sparsity_threshold = 0.05f;

// Balanced (good accuracy/speed tradeoff)
params.sparsity_threshold = 0.1f;   // Default

// Aggressive (higher speedup, may impact accuracy)
params.sparsity_threshold = 0.2f;
```

### Architecture-Specific Optimizations

```cpp
// Force specific architecture if auto-detection fails
params.auto_detect_arch = false;
params.arch_override = "llama";  // or "mistral", "phi", "gemma"

// Enable advanced optimizations for supported architectures
params.optimization_level = 2.0f;  // Maximum optimizations
```

## Troubleshooting

### Common Issues

#### 1. Initialization Fails
```
Error: ATLAS initialization failed: Architecture not supported
```
**Solution:** Check model architecture and ensure it's supported, or disable auto-detection:
```cpp
params.auto_detect_arch = false;
params.arch_override = "llama";  // Use generic Llama optimizations
```

#### 2. Memory Pool Exhausted
```
Error: ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED
```
**Solution:** Increase memory pool size:
```cpp
params.memory_pool_size = 512 * 1024 * 1024;  // 512MB
```

#### 3. Performance Degradation
**Symptoms:** ATLAS enabled but inference is slower
**Solution:** Check sparsity threshold and architecture detection:
```cpp
atlas_runtime_stats stats;
atlas_runtime_get_stats(runtime_ctx, &stats);
printf("ATLAS speedup: %.2fx\n", stats.average_speedup);

if (stats.average_speedup < 1.0f) {
    // Disable ATLAS for this model
    atlas_runtime_disable_for_context(runtime_ctx, ctx);
}
```

#### 4. Memory Leaks
**Symptoms:** Memory usage increases over time
**Solution:** Ensure proper cleanup:
```cpp
// Always call cleanup
atlas_runtime_cleanup(runtime_ctx);

// Check for proper resource cleanup
atlas_runtime_stats stats;
atlas_runtime_get_stats(runtime_ctx, &stats);
if (stats.total_memory_used > 0) {
    fprintf(stderr, "Warning: ATLAS memory not fully freed\n");
}
```

### Debug Mode

Enable detailed logging for troubleshooting:

```cpp
atlas_runtime_params params = atlas_runtime_default_params();
params.debug_level = ATLAS_DEBUG_VERBOSE;
params.log_file = "atlas_debug.log";
```

### Performance Profiling

```cpp
// Enable performance counters
atlas_runtime_stats stats;
atlas_runtime_get_stats(runtime_ctx, &stats);

printf("ATLAS Performance Summary:\n");
printf("  Total memory used: %zu MB\n", stats.total_memory_used / (1024 * 1024));
printf("  Modules loaded: %zu\n", stats.modules_loaded);
printf("  Average speedup: %.2fx\n", stats.average_speedup);
printf("  Operations processed: %lu\n", stats.operations_processed);
```

## Examples

### Example 1: Simple Chat Application

```cpp
#include "atlas-runtime.h"
#include "llama.h"
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 2) {
        printf("Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    
    // Initialize backends
    llama_backend_init();
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(argv[1], model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Initialize ATLAS
    atlas_runtime_params atlas_params = atlas_runtime_default_params();
    atlas_params.memory_pool_size = 128 * 1024 * 1024;  // 128MB
    atlas_params.auto_detect_arch = true;
    
    atlas_runtime_ctx * runtime_ctx = nullptr;
    atlas_runtime_result result = atlas_runtime_init(&runtime_ctx, model, &atlas_params);
    
    if (result != ATLAS_RUNTIME_SUCCESS) {
        fprintf(stderr, "ATLAS initialization failed: %s\n", 
                atlas_runtime_error_string(result));
        // Continue without ATLAS
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        atlas_runtime_cleanup(runtime_ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    
    // Enable ATLAS for context
    if (runtime_ctx) {
        atlas_runtime_enable_for_context(runtime_ctx, ctx);
    }
    
    // Chat loop
    std::string input;
    printf("Chat with ATLAS-optimized model (type 'quit' to exit):\n> ");
    
    while (std::getline(std::cin, input) && input != "quit") {
        // Tokenize input
        std::vector<llama_token> tokens = llama_tokenize(ctx, input, true);
        
        // Process tokens
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (llama_eval(ctx, &tokens[i], 1, i, 1) != 0) {
                fprintf(stderr, "llama_eval failed\n");
                break;
            }
        }
        
        // Generate response (simplified)
        for (int i = 0; i < 50; ++i) {  // Generate up to 50 tokens
            llama_token next_token = llama_sample_token_greedy(ctx, nullptr);
            if (next_token == llama_token_eos(model)) break;
            
            printf("%s", llama_token_to_piece(ctx, next_token).c_str());
            fflush(stdout);
            
            if (llama_eval(ctx, &next_token, 1, tokens.size() + i, 1) != 0) {
                fprintf(stderr, "llama_eval failed\n");
                break;
            }
        }
        
        printf("\n> ");
    }
    
    // Show ATLAS statistics
    if (runtime_ctx) {
        atlas_runtime_stats stats;
        atlas_runtime_get_stats(runtime_ctx, &stats);
        printf("\nATLAS Performance Summary:\n");
        printf("  Average speedup: %.2fx\n", stats.average_speedup);
        printf("  Memory used: %zu MB\n", stats.total_memory_used / (1024 * 1024));
    }
    
    // Cleanup
    atlas_runtime_cleanup(runtime_ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}
```

### Example 2: Batch Processing

```cpp
#include "atlas-runtime.h"
#include "llama.h"
#include <vector>
#include <chrono>

struct batch_item {
    std::string text;
    std::vector<float> embeddings;
};

int main(int argc, char ** argv) {
    if (argc != 3) {
        printf("Usage: %s <model.gguf> <input.txt>\n", argv[0]);
        return 1;
    }
    
    // Load model with ATLAS
    llama_backend_init();
    llama_model * model = llama_load_model_from_file(argv[1], llama_model_default_params());
    
    atlas_runtime_params atlas_params = atlas_runtime_default_params();
    atlas_params.memory_pool_size = 256 * 1024 * 1024;  // 256MB for batch processing
    atlas_params.optimization_level = 1.5f;  // Higher optimization for batch work
    
    atlas_runtime_ctx * runtime_ctx = nullptr;
    atlas_runtime_init(&runtime_ctx, model, &atlas_params);
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 512;  // Larger batch size
    
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    atlas_runtime_enable_for_context(runtime_ctx, ctx);
    
    // Read input file
    std::vector<batch_item> items;
    // ... load items from file ...
    
    // Process batch
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (auto & item : items) {
        std::vector<llama_token> tokens = llama_tokenize(ctx, item.text, true);
        
        // Process in batches
        for (size_t i = 0; i < tokens.size(); i += ctx_params.n_batch) {
            size_t batch_size = std::min(ctx_params.n_batch, tokens.size() - i);
            llama_eval(ctx, &tokens[i], batch_size, i, 1);
        }
        
        // Extract embeddings
        const float * embeddings = llama_get_embeddings(ctx);
        item.embeddings.assign(embeddings, embeddings + llama_n_embd(model));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    printf("Processed %zu items in %ld ms\n", items.size(), duration.count());
    
    // Show ATLAS performance
    atlas_runtime_stats stats;
    atlas_runtime_get_stats(runtime_ctx, &stats);
    printf("ATLAS speedup: %.2fx\n", stats.average_speedup);
    
    // Cleanup
    atlas_runtime_cleanup(runtime_ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}
```

### Example 3: Server Application

```cpp
#include "atlas-runtime.h"
#include "llama.h"
#include <thread>
#include <mutex>
#include <vector>

class atlas_llama_server {
private:
    llama_model * model;
    atlas_runtime_ctx * runtime_ctx;
    std::vector<llama_context *> contexts;
    std::mutex contexts_mutex;
    
public:
    bool initialize(const std::string & model_path, int num_contexts = 4) {
        // Load model
        llama_backend_init();
        model = llama_load_model_from_file(model_path.c_str(), llama_model_default_params());
        if (!model) return false;
        
        // Initialize ATLAS
        atlas_runtime_params atlas_params = atlas_runtime_default_params();
        atlas_params.memory_pool_size = 512 * 1024 * 1024;  // 512MB for server
        atlas_params.max_modules = 32;  // More modules for server
        
        atlas_runtime_result result = atlas_runtime_init(&runtime_ctx, model, &atlas_params);
        if (result != ATLAS_RUNTIME_SUCCESS) {
            fprintf(stderr, "ATLAS initialization failed\n");
            return false;
        }
        
        // Create contexts
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        
        for (int i = 0; i < num_contexts; ++i) {
            llama_context * ctx = llama_new_context_with_model(model, ctx_params);
            if (ctx) {
                atlas_runtime_enable_for_context(runtime_ctx, ctx);
                contexts.push_back(ctx);
            }
        }
        
        return !contexts.empty();
    }
    
    llama_context * get_context() {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        if (contexts.empty()) return nullptr;
        
        llama_context * ctx = contexts.back();
        contexts.pop_back();
        return ctx;
    }
    
    void return_context(llama_context * ctx) {
        std::lock_guard<std::mutex> lock(contexts_mutex);
        contexts.push_back(ctx);
    }
    
    std::string process_request(const std::string & input) {
        llama_context * ctx = get_context();
        if (!ctx) return "Error: No available contexts";
        
        // Process request
        std::vector<llama_token> tokens = llama_tokenize(ctx, input, true);
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            llama_eval(ctx, &tokens[i], 1, i, 1);
        }
        
        // Generate response
        std::string response;
        for (int i = 0; i < 100; ++i) {
            llama_token next_token = llama_sample_token_greedy(ctx, nullptr);
            if (next_token == llama_token_eos(model)) break;
            
            response += llama_token_to_piece(ctx, next_token);
            llama_eval(ctx, &next_token, 1, tokens.size() + i, 1);
        }
        
        return_context(ctx);
        return response;
    }
    
    void shutdown() {
        for (llama_context * ctx : contexts) {
            llama_free(ctx);
        }
        contexts.clear();
        
        atlas_runtime_cleanup(runtime_ctx);
        llama_free_model(model);
        llama_backend_free();
    }
    
    void print_stats() {
        atlas_runtime_stats stats;
        atlas_runtime_get_stats(runtime_ctx, &stats);
        
        printf("ATLAS Server Statistics:\n");
        printf("  Memory used: %zu MB\n", stats.total_memory_used / (1024 * 1024));
        printf("  Modules loaded: %zu\n", stats.modules_loaded);
        printf("  Average speedup: %.2fx\n", stats.average_speedup);
        printf("  Total operations: %lu\n", stats.operations_processed);
    }
};
```

## Conclusion

The ATLAS Runtime system provides a powerful way to optimize GGUF models without requiring file conversion. By detecting model architecture and applying appropriate optimizations dynamically, ATLAS can significantly improve inference performance while maintaining compatibility with existing llama.cpp workflows.

For additional support and advanced configurations, refer to the API documentation or check the test files for more usage examples.