## Summary

Implement seamless ATLAS support for existing GGUF models without requiring model conversion. Enable any standard GGUF model to use ATLAS test-time memorization through runtime parameter initialization and dynamic memory module creation.

## Background

Users should be able to load any existing GGUF model and enable ATLAS capabilities without:
- Pre-converting models to ATLAS format
- Modifying original model files
- Requiring ATLAS-specific model training

ATLAS memory modules and optimization components should be initialized at runtime, allowing standard models to gain test-time memorization capabilities instantly.

## Implementation Requirements

### 1. Runtime ATLAS Module Initialization

#### Dynamic Memory Module Creation
```c
// Initialize ATLAS components for any GGUF model
int atlas_initialize_for_model(
    struct atlas_context* atlas_ctx,
    const struct llama_model* model,
    const struct atlas_init_params* params
);

struct atlas_init_params {
    // Memory module configuration
    int memory_hidden_dim;      // Hidden dimension for 2-layer MLP
    int memory_layers;          // Which layers to add ATLAS (-1 = all)
    
    // Sliding window configuration
    int window_size;            // Omega Rule sliding window size
    float temporal_decay;       // Exponential decay for window weights
    
    // Feature mapping configuration
    enum atlas_feature_type feature_type;  // Polynomial, exponential, etc.
    int polynomial_degree;      // Degree for polynomial kernels
    
    // Optimization configuration
    bool use_muon_optimizer;    // Use Muon vs standard SGD
    float learning_rate;        // Test-time learning rate
    int muon_iterations;        // Newton-Schulz iterations
    
    // Blending configuration
    float initial_blend_ratio;  // Starting ATLAS/standard attention mix
    bool adaptive_blending;     // Enable adaptive blend ratio
    
    // Initialization strategy
    enum atlas_init_strategy {
        ATLAS_INIT_RANDOM,      // Random initialization
        ATLAS_INIT_XAVIER,      // Xavier/Glorot initialization
        ATLAS_INIT_IDENTITY,    // Initialize as identity mapping
        ATLAS_INIT_PRETRAINED   // Load from pretrained ATLAS weights
    } init_strategy;
};

// Per-layer ATLAS module initialization
int atlas_init_layer_modules(
    struct atlas_layer_context* layer_ctx,
    int layer_idx,
    int input_dim,
    int hidden_dim,
    enum atlas_init_strategy strategy
);
```

#### Model Compatibility Detection
```c
// Analyze model architecture for ATLAS compatibility
struct atlas_compatibility_info {
    bool is_compatible;
    int n_layers;
    int n_embd;
    int n_head;
    int n_head_kv;
    char architecture[64];      // "llama", "mistral", etc.
    
    // Recommended ATLAS settings
    int recommended_memory_dim;
    int recommended_window_size;
    float recommended_blend_ratio;
    
    // Compatibility warnings
    char warnings[1024];
};

int atlas_analyze_model_compatibility(
    const struct llama_model* model,
    struct atlas_compatibility_info* info
);

// Automatic parameter selection based on model
int atlas_auto_configure_params(
    const struct llama_model* model,
    struct atlas_init_params* params
);
```

### 2. Zero-Conversion Model Loading

#### Enhanced Model Loading Pipeline
```c
// Extended llama_load_model_from_file with ATLAS support
struct llama_model* llama_load_model_from_file_atlas(
    const char* path_model,
    struct llama_model_params params,
    struct atlas_init_params* atlas_params  // NULL = no ATLAS
);

// Retrofit existing model with ATLAS
int llama_model_add_atlas_support(
    struct llama_model* model,
    const struct atlas_init_params* atlas_params
);

// Check if model has ATLAS support
bool llama_model_has_atlas(const struct llama_model* model);

// Runtime ATLAS configuration for existing model
int llama_model_configure_atlas(
    struct llama_model* model,
    const struct atlas_runtime_config* config
);

struct atlas_runtime_config {
    bool enabled;
    float blend_ratio;
    int window_size;
    float learning_rate;
    bool save_memory_on_exit;
    char memory_file[512];
};
```

#### Memory-Efficient Initialization
```c
// Initialize ATLAS modules on-demand during first use
struct atlas_lazy_init {
    bool modules_initialized;
    struct atlas_init_params init_params;
    void* initialization_data;
    
    // Lazy initialization function
    int (*init_func)(struct atlas_context*, const struct llama_model*);
};

// On-demand initialization during first inference
int atlas_ensure_initialized(
    struct atlas_context* atlas_ctx,
    const struct llama_model* model
) {
    if (!atlas_ctx->lazy_init.modules_initialized) {
        return atlas_ctx->lazy_init.init_func(atlas_ctx, model);
    }
    return 0;
}
```

### 3. Dynamic Parameter Adjustment

#### Runtime Parameter Modification
```c
// Modify ATLAS parameters during inference
int atlas_update_runtime_params(
    struct atlas_context* atlas_ctx,
    const struct atlas_runtime_update* update
);

struct atlas_runtime_update {
    // Optional parameter updates (set to -1 to keep current value)
    float blend_ratio;          // New blend ratio
    int window_size;           // New sliding window size
    float learning_rate;       // New optimization learning rate
    bool enable_muon;          // Toggle Muon optimizer
    int polynomial_degree;     // Change feature mapping degree
    
    // Behavioral flags
    bool reset_memory;         // Reset memory modules to initial state
    bool reset_optimizer;      // Reset Muon optimizer state
    bool clear_window;         // Clear sliding window context
};

// Adaptive parameter adjustment based on performance
struct atlas_adaptive_config {
    bool enable_adaptive_blend;     // Adjust blend ratio automatically
    bool enable_adaptive_window;    // Adjust window size automatically
    bool enable_adaptive_lr;        // Adjust learning rate automatically
    
    // Adaptation criteria
    float perplexity_threshold;     // Trigger adaptation if perplexity > threshold
    float improvement_threshold;    // Min improvement to maintain settings
    int adaptation_interval;        // Evaluate every N tokens
};

int atlas_configure_adaptive_behavior(
    struct atlas_context* atlas_ctx,
    const struct atlas_adaptive_config* config
);
```

### 4. Model Architecture Adaptation

#### Architecture-Specific Optimizations
```c
// Architecture-specific ATLAS configurations
struct atlas_arch_config {
    char architecture[64];      // "llama", "mistral", "phi", etc.
    
    // Optimal parameters for this architecture
    int memory_dim_multiplier;  // Multiply n_embd by this factor
    float recommended_blend_ratio;
    int recommended_window_size;
    enum atlas_feature_type preferred_feature_type;
    
    // Architecture-specific hooks
    int (*custom_init_func)(struct atlas_context*, const struct llama_model*);
    int (*custom_forward_func)(struct atlas_context*, struct ggml_tensor*);
};

// Register architecture-specific configurations
int atlas_register_architecture_config(const struct atlas_arch_config* config);

// Get optimal configuration for model architecture
const struct atlas_arch_config* atlas_get_arch_config(const char* architecture);

// Built-in configurations for popular architectures
extern const struct atlas_arch_config atlas_llama_config;
extern const struct atlas_arch_config atlas_mistral_config;
extern const struct atlas_arch_config atlas_phi_config;
extern const struct atlas_arch_config atlas_gemma_config;
```

### 5. Command-Line Interface Enhancements

#### Simplified CLI Usage
```bash
# Enable ATLAS for any GGUF model with defaults
./llama-cli -m any-model.gguf --atlas

# Enable ATLAS with custom parameters
./llama-cli -m model.gguf \
    --atlas \
    --atlas-memory-dim 2048 \
    --atlas-window 1024 \
    --atlas-blend 0.6 \
    --atlas-polynomial 3

# Auto-configure ATLAS based on model
./llama-cli -m model.gguf --atlas-auto

# Enable ATLAS with memory persistence
./llama-cli -m model.gguf \
    --atlas \
    --atlas-memory-file session.atlas \
    --atlas-auto-save

# Server with automatic ATLAS detection
./llama-server -m model.gguf --atlas-auto --host 0.0.0.0 --port 8080
```

#### Parameter Auto-Detection
```bash
# Display recommended ATLAS parameters for model
./llama-inspect-atlas -m model.gguf

# Output:
# Model: llama-7b-chat.gguf
# Architecture: llama
# Recommended ATLAS settings:
#   --atlas-memory-dim 2048
#   --atlas-window 512  
#   --atlas-blend 0.5
#   --atlas-polynomial 3
#   --atlas-muon
# 
# Estimated memory overhead: 1.2GB
# Recommended for contexts: >4K tokens
```

### 6. Integration with Existing Workflows

#### Backward Compatibility
```c
// Ensure all existing APIs work unchanged
struct llama_context* llama_new_context_with_model(
    struct llama_model* model,
    struct llama_context_params params
) {
    // Initialize context normally
    struct llama_context* ctx = llama_new_context_impl(model, params);
    
    // Add ATLAS if requested in params
    if (params.atlas_enabled) {
        atlas_initialize_for_model(&ctx->atlas_ctx, model, &params.atlas_params);
    }
    
    return ctx;
}

// Extended context parameters with ATLAS support
struct llama_context_params {
    // ... existing parameters ...
    
    // ATLAS parameters (optional)
    bool atlas_enabled;
    struct atlas_init_params atlas_params;
    char atlas_memory_file[512];
    bool atlas_auto_save;
};
```

#### API Compatibility Layer
```c
// Compatibility functions for existing code
#define llama_new_context_with_model(model, params) \
    llama_new_context_with_model_atlas(model, params, NULL)

// Enhanced version with ATLAS support
struct llama_context* llama_new_context_with_model_atlas(
    struct llama_model* model,
    struct llama_context_params params,
    struct atlas_init_params* atlas_params
);

// Runtime ATLAS enable/disable
int llama_enable_atlas(struct llama_context* ctx, const struct atlas_init_params* params);
int llama_disable_atlas(struct llama_context* ctx);
bool llama_is_atlas_enabled(const struct llama_context* ctx);
```

## Testing Requirements

### Model Compatibility Tests
- [ ] **Architecture support**: Test with Llama, Mistral, Phi, Gemma models
- [ ] **Size compatibility**: Test from 1B to 70B parameter models
- [ ] **Quantization support**: Test with Q4, Q5, Q8 quantized models
- [ ] **Context length**: Test with various context length models
- [ ] **Fine-tuned models**: Test with instruction-tuned and chat models

### Initialization Tests
- [ ] **Parameter auto-detection**: Verify optimal parameters for each architecture
- [ ] **Memory allocation**: Ensure proper memory allocation for ATLAS modules
- [ ] **Initialization strategies**: Test random, Xavier, identity initialization
- [ ] **Runtime configuration**: Test parameter changes during inference

### Performance Tests
- [ ] **Initialization overhead**: ATLAS initialization adds <5 seconds
- [ ] **Memory overhead**: Memory usage increase within expected bounds
- [ ] **Inference impact**: Minimal performance impact when ATLAS disabled
- [ ] **Model loading time**: ATLAS-enabled loading <20% slower than standard

## Implementation Files

### Core Runtime Initialization
- `src/atlas/atlas-runtime-init.h` - Runtime initialization API
- `src/atlas/atlas-runtime-init.cpp` - Core initialization implementation
- `src/atlas/atlas-model-compat.cpp` - Model compatibility analysis

### Architecture Support
- `src/atlas/arch/atlas-llama.cpp` - Llama architecture optimizations
- `src/atlas/arch/atlas-mistral.cpp` - Mistral architecture optimizations
- `src/atlas/arch/atlas-generic.cpp` - Generic architecture fallback

### CLI Integration
- `examples/cli/atlas-cli-runtime.cpp` - CLI ATLAS runtime support
- `examples/server/atlas-server-runtime.cpp` - Server ATLAS runtime support
- `tools/atlas-inspector.cpp` - Model analysis tool

### Test Files
- `tests/atlas/test-runtime-init.cpp` - Runtime initialization tests
- `tests/atlas/test-model-compat.cpp` - Model compatibility tests
- `tests/atlas/test-arch-support.cpp` - Architecture-specific tests

## Success Criteria

### Functional Requirements
- [ ] Any GGUF model can enable ATLAS without modification
- [ ] Automatic parameter detection works for all supported architectures
- [ ] Runtime parameter adjustment functions correctly
- [ ] Zero impact on models when ATLAS disabled
- [ ] Full backward compatibility with existing API

### Performance Requirements
- [ ] ATLAS initialization completes in <5 seconds for 7B models
- [ ] Memory overhead within documented bounds (typically +30%)
- [ ] No performance regression for non-ATLAS inference
- [ ] Parameter auto-detection executes in <1 second

### Quality Requirements
- [ ] 100% compatibility with existing llama.cpp workflows
- [ ] Robust error handling for unsupported model formats
- [ ] Clear documentation for optimal parameter selection
- [ ] Comprehensive testing across model architectures

## Architecture-Specific Optimizations

### Llama Family Models
```c
const struct atlas_arch_config atlas_llama_config = {
    .architecture = "llama",
    .memory_dim_multiplier = 2,     // memory_dim = 2 * n_embd
    .recommended_blend_ratio = 0.5,
    .recommended_window_size = 512,
    .preferred_feature_type = ATLAS_FEATURE_POLYNOMIAL,
    .custom_init_func = atlas_llama_init,
    .custom_forward_func = NULL     // Use default
};
```

### Mistral Family Models  
```c
const struct atlas_arch_config atlas_mistral_config = {
    .architecture = "mistral",
    .memory_dim_multiplier = 1,     // memory_dim = n_embd
    .recommended_blend_ratio = 0.7,
    .recommended_window_size = 1024,
    .preferred_feature_type = ATLAS_FEATURE_EXPONENTIAL,
    .custom_init_func = atlas_mistral_init,
    .custom_forward_func = atlas_mistral_forward
};
```

## Dependencies
- Issues #3-12: All ATLAS components and persistence
- GGML model loading infrastructure
- Memory allocation and management systems
- Command-line parameter parsing

## Estimated Effort
**2-3 weeks** for experienced llama.cpp developer

## Usage Examples

### Simple ATLAS Enablement
```bash
# Any model gets ATLAS capabilities
./llama-cli -m llama-7b-chat.gguf --atlas -p "Long context prompt..."
./llama-cli -m mistral-7b.gguf --atlas -p "Analysis task..."
./llama-cli -m phi-3.gguf --atlas -p "Question answering..."
```

### Advanced Configuration
```bash
# Custom ATLAS parameters
./llama-server -m any-model.gguf \
    --atlas \
    --atlas-memory-dim 4096 \
    --atlas-window 2048 \
    --atlas-blend 0.8 \
    --atlas-memory-file domain_specific.atlas
```

### Architecture Auto-Detection
```bash
# Let ATLAS choose optimal parameters
./llama-cli -m unknown-model.gguf --atlas-auto

# Inspect recommended parameters
./llama-inspect-atlas -m model.gguf --show-recommendations
```

## References
- GGML model format documentation
- llama.cpp architecture overview
- Memory allocation best practices
- Dynamic initialization patterns
