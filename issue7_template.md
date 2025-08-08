## Summary

Integrate all ATLAS components into the llama.cpp inference pipeline, enabling ATLAS-enhanced attention mechanisms and long-context processing. This phase connects ATLAS components with existing llama.cpp architecture.

## Background

This integration phase transforms llama.cpp from traditional O(n²) attention to ATLAS's O(w·n) linear scaling while maintaining full backward compatibility with existing models and functionality.

## Implementation Requirements

### 1. Enhanced Attention Integration
```c
// Enhanced attention function signature
struct ggml_tensor * llm_build_atlas_attention(
    struct ggml_context * ctx,
    struct llama_context * lctx,
    struct ggml_tensor * cur,              // Current input tensor
    struct atlas_context * atlas_ctx,      // ATLAS context
    const llama_batch & batch,
    int n_tokens,
    int kv_head,
    int n_kv,
    float max_alibi_bias
);

// Modified llm_build_llama function
static struct ggml_cgraph * llm_build_llama(
    llama_context & lctx,
    const llama_batch & batch) {
    
    // ... existing code ...
    
#ifdef GGML_USE_ATLAS
    if (lctx.atlas_enabled) {
        // Apply ATLAS-enhanced attention
        cur = llm_build_atlas_attention(ctx0, &lctx, cur, 
                                       &lctx.atlas_ctx, batch,
                                       n_tokens, kv_head, n_kv, 
                                       max_alibi_bias);
    } else {
        // Standard attention path
        cur = llm_build_kv_store(ctx0, hparams, /*...*/);
    }
#else
    // Standard attention path (when ATLAS disabled)
    cur = llm_build_kv_store(ctx0, hparams, /*...*/);
#endif
    
    // ... rest of existing code ...
}
```

### 2. Context Management Integration
```c
// Extend llama_context with ATLAS support
struct llama_context {
    // ... existing fields ...
    
#ifdef GGML_USE_ATLAS
    bool atlas_enabled;                    // ATLAS feature flag
    struct atlas_context atlas_ctx;       // ATLAS context
    struct atlas_memory_pool * atlas_pool; // Dedicated ATLAS memory
    float atlas_blend_ratio;              // Blend ATLAS + standard attention
    int atlas_warmup_tokens;              // Tokens before ATLAS activation
#endif
};

// ATLAS context initialization
int llama_atlas_init(
    struct llama_context * ctx,
    const struct llama_context_params * params
);

// ATLAS context cleanup
void llama_atlas_free(struct llama_context * ctx);
```

### 3. Model Loading Extensions (GGUF)
```c
// Extended GGUF keys for ATLAS models
enum gguf_keys {
    // ... existing keys ...
    GGUF_KEY_ATLAS_ENABLED,
    GGUF_KEY_ATLAS_MEMORY_DIM,
    GGUF_KEY_ATLAS_WINDOW_SIZE,
    GGUF_KEY_ATLAS_MUON_ITERATIONS,
    GGUF_KEY_ATLAS_FEATURE_TYPE,
    GGUF_KEY_ATLAS_OMEGA_ALPHA,
    GGUF_KEY_ATLAS_BLEND_RATIO,
};

// ATLAS tensor names in GGUF
#define ATLAS_TENSOR_MEMORY_W1      "atlas.memory.weight1"
#define ATLAS_TENSOR_MEMORY_B1      "atlas.memory.bias1"
#define ATLAS_TENSOR_MEMORY_W2      "atlas.memory.weight2"
#define ATLAS_TENSOR_MEMORY_B2      "atlas.memory.bias2"
#define ATLAS_TENSOR_MEMORY_RESIDUAL "atlas.memory.residual"
#define ATLAS_TENSOR_FEATURE_WEIGHTS "atlas.feature.weights"
#define ATLAS_TENSOR_MUON_MOMENTUM  "atlas.muon.momentum"

// Model loading with ATLAS support
static bool llama_model_load_tensors(
    struct llama_model_loader & ml,
    llama_model & model) {
    
    // ... existing tensor loading ...
    
#ifdef GGML_USE_ATLAS
    // Check for ATLAS metadata
    int atlas_enabled_key = gguf_find_key(ml.ctx_gguf, "atlas.enabled");
    if (atlas_enabled_key != -1) {
        model.atlas_enabled = gguf_get_val_bool(ml.ctx_gguf, atlas_enabled_key);
        
        if (model.atlas_enabled) {
            // Load ATLAS-specific hyperparameters
            model.hparams.atlas_memory_dim = 
                gguf_get_val_u32(ml.ctx_gguf, gguf_find_key(ml.ctx_gguf, "atlas.memory_dim"));
            model.hparams.atlas_window_size =
                gguf_get_val_u32(ml.ctx_gguf, gguf_find_key(ml.ctx_gguf, "atlas.window_size"));
            
            // Load ATLAS tensors
            atlas_load_model_tensors(ml, model);
        }
    }
#endif
    
    return true;
}
```

### 4. Enhanced Attention Implementation
```c
struct ggml_tensor * llm_build_atlas_attention(
    struct ggml_context * ctx,
    struct llama_context * lctx,
    struct ggml_tensor * cur,
    struct atlas_context * atlas_ctx,
    const llama_batch & batch,
    int n_tokens, int kv_head, int n_kv,
    float max_alibi_bias) {
    
    const auto & hparams = lctx->model.hparams;
    const int n_embd = hparams.n_embd;
    const int n_head = hparams.n_head;
    const int n_head_kv = hparams.n_head_kv;
    
    // Standard Q/K/V projections
    struct ggml_tensor * q = ggml_mul_mat(ctx, lctx->model.layers[il].wq, cur);
    struct ggml_tensor * k = ggml_mul_mat(ctx, lctx->model.layers[il].wk, cur);
    struct ggml_tensor * v = ggml_mul_mat(ctx, lctx->model.layers[il].wv, cur);
    
    // Apply RoPE if enabled
    q = llm_build_rope(ctx, lctx, q, /*...*/);
    k = llm_build_rope(ctx, lctx, k, /*...*/);
    
    // Apply feature mapping to keys
    struct ggml_tensor * k_enhanced = atlas_apply_feature_mapping(
        ctx, k, &atlas_ctx->feature_mapper);
    
    // Update sliding window with new key-value pair
    atlas_window_add_kv(&atlas_ctx->omega.window, k_enhanced, v, n_tokens);
    
    // Compute ATLAS memory-enhanced attention
    struct ggml_tensor * atlas_output = atlas_memory_forward(
        ctx, &atlas_ctx->memory, k_enhanced);
    
    // Update memory using Omega rule (if optimization enabled)
    if (atlas_ctx->enable_optimization) {
        struct ggml_tensor * omega_loss = atlas_omega_loss(
            ctx, &atlas_ctx->omega, &atlas_ctx->memory, n_tokens);
        
        // Apply Muon optimizer update
        atlas_muon_step(&atlas_ctx->muon, atlas_ctx->memory.w1, 
                       ggml_grad(omega_loss), ctx);
    }
    
    // Compute traditional attention for blending
    struct ggml_tensor * standard_attn = ggml_flash_attn_ext(
        ctx, q, k, v, kv_mask, n_kv, max_alibi_bias, 
        hparams.f_max_alibi_bias, attn_factor);
    
    // Blend ATLAS and standard attention
    float blend_ratio = atlas_ctx->blend_ratio;
    struct ggml_tensor * blended = ggml_add(ctx,
        ggml_scale(ctx, atlas_output, blend_ratio),
        ggml_scale(ctx, standard_attn, 1.0f - blend_ratio));
    
    return blended;
}
```

### 5. Context Parameter Extensions
```c
// Extended context parameters for ATLAS
struct llama_context_params {
    // ... existing parameters ...
    
#ifdef GGML_USE_ATLAS
    bool atlas_enabled;              // Enable ATLAS processing
    float atlas_blend_ratio;         // ATLAS vs standard attention (0.0-1.0)
    int atlas_warmup_tokens;         // Tokens before ATLAS activation
    bool atlas_adaptive_window;      // Enable adaptive window sizing
    float atlas_memory_budget_mb;    // Memory budget for ATLAS (MB)
    enum atlas_feature_type atlas_feature_type; // Feature mapping type
    bool atlas_enable_optimization;  // Enable test-time optimization
#endif
};

// Default ATLAS parameters
struct llama_context_params llama_context_default_params() {
    struct llama_context_params result = {
        // ... existing defaults ...
        
#ifdef GGML_USE_ATLAS
        /*.atlas_enabled              =*/ false,
        /*.atlas_blend_ratio          =*/ 0.1f,
        /*.atlas_warmup_tokens        =*/ 512,
        /*.atlas_adaptive_window      =*/ true,
        /*.atlas_memory_budget_mb     =*/ 300.0f,
        /*.atlas_feature_type         =*/ ATLAS_FEATURE_POLYNOMIAL,
        /*.atlas_enable_optimization  =*/ true,
#endif
    };
    return result;
}
```

### 6. API Extensions
```c
// Check if model supports ATLAS
LLAMA_API bool llama_model_has_atlas(const struct llama_model * model);

// Enable/disable ATLAS at runtime
LLAMA_API void llama_set_atlas_enabled(struct llama_context * ctx, bool enabled);

// Get ATLAS statistics
struct llama_atlas_stats {
    float memory_usage_mb;           // Current ATLAS memory usage
    int window_fill_ratio;           // Sliding window utilization %
    float optimization_loss;         // Current Omega rule loss
    int muon_iterations_completed;   // Total Muon optimization steps
    float cache_hit_ratio;           // Feature cache hit rate
};

LLAMA_API struct llama_atlas_stats llama_get_atlas_stats(const struct llama_context * ctx);

// Configure ATLAS parameters at runtime
LLAMA_API void llama_set_atlas_blend_ratio(struct llama_context * ctx, float ratio);
LLAMA_API void llama_set_atlas_feature_type(struct llama_context * ctx, enum atlas_feature_type type);
```

## Testing Requirements

### Integration Tests
- [ ] **Full pipeline testing**: ATLAS attention integrated with existing llama.cpp flow
- [ ] **Model compatibility**: Test with various model sizes (7B, 13B, 70B)
- [ ] **Backward compatibility**: Ensure non-ATLAS models work unchanged
- [ ] **Context length scaling**: Test long contexts (8K, 32K, 128K tokens)
- [ ] **Multi-sequence batching**: Validate batch processing with ATLAS

### Performance Tests
- [ ] **Throughput comparison**: ATLAS vs standard attention across context lengths
- [ ] **Memory usage**: Monitor memory consumption with various configurations
- [ ] **Latency analysis**: Per-token latency with ATLAS enabled
- [ ] **Scaling behavior**: Performance vs context length curves

### Functionality Tests
- [ ] **Model loading**: GGUF models with/without ATLAS metadata
- [ ] **Parameter validation**: Test all ATLAS configuration parameters
- [ ] **Runtime switching**: Enable/disable ATLAS during inference
- [ ] **Blend ratio effects**: Validate attention blending at different ratios

### Quality Tests
- [ ] **Output quality**: Perplexity and generation quality with ATLAS
- [ ] **Numerical stability**: Long sequences without degradation
- [ ] **Error handling**: Graceful fallback when ATLAS fails
- [ ] **Memory leaks**: Extended runs without memory growth

## Implementation Files

### Core Integration
- `src/llama-atlas.cpp` - Main ATLAS integration implementation
- `src/llama-atlas.h` - ATLAS-specific API declarations
- `include/llama-atlas.h` - Public API for ATLAS functionality

### Model Loading
- `src/llama-model-atlas.cpp` - GGUF loading with ATLAS support
- `src/llama-tensor-atlas.cpp` - ATLAS tensor management

### Enhanced Attention
- `src/llama-attention-atlas.cpp` - ATLAS-enhanced attention implementation
- `src/llama-kv-atlas.cpp` - ATLAS KV cache management

### Examples and Tools
- `examples/atlas-inference/` - Example ATLAS inference application
- `examples/atlas-benchmark/` - Performance benchmarking tool
- `tools/atlas-convert/` - Convert standard models to ATLAS format

### Test Files
- `tests/test-llama-atlas.cpp` - Integration tests
- `tests/test-atlas-attention.cpp` - Attention mechanism tests
- `tests/test-atlas-performance.cpp` - Performance validation

## Success Criteria

### Functional Requirements
- [ ] ATLAS attention produces coherent outputs for all model sizes
- [ ] Context lengths up to 128K tokens process successfully
- [ ] Backward compatibility maintained for all existing functionality
- [ ] API extensions work correctly and don't break existing code

### Performance Requirements
- [ ] Linear scaling demonstrated for contexts >8K tokens
- [ ] Memory usage <300MB overhead for typical configurations
- [ ] Throughput degradation <25% for short contexts (<2K tokens)
- [ ] Throughput improvement >200% for long contexts (>32K tokens)

### Quality Requirements
- [ ] Output perplexity within 5% of baseline for standard benchmarks
- [ ] No generation quality degradation for common use cases
- [ ] Stable inference for extended sequences (1M+ tokens theoretical)
- [ ] Robust error handling and recovery

## Dependencies
- Issue #3: Deep Memory Module
- Issue #4: Omega Rule Implementation  
- Issue #5: Muon Optimizer
- Issue #6: Feature Mapping
- GGUF format extensions
- CUDA backend infrastructure

## Performance Targets
- **Context scaling**: Linear time complexity for contexts >8K tokens
- **Memory efficiency**: <300MB fixed overhead regardless of context length
- **Throughput**: Match standard attention for <2K contexts, exceed for >8K
- **Quality**: Maintain perplexity within 5% of baseline models

## Advanced Features
- [ ] **Dynamic blend ratio**: Automatically adjust ATLAS/standard attention mix
- [ ] **Context-aware optimization**: Adaptive parameters based on content
- [ ] **Multi-device support**: ATLAS across multiple GPUs
- [ ] **Quantization compatibility**: ATLAS with INT8/INT4 quantized models

## Estimated Effort
**3 weeks** for experienced llama.cpp developer with ATLAS component familiarity

## References
- llama.cpp architecture documentation
- GGUF format specification
- ATLAS research paper integration guidelines
- Long-context attention mechanisms
