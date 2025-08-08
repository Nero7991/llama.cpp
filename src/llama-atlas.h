#pragma once

#include "ggml.h"
#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS Configuration Structure
struct atlas_config {
    // Global ATLAS settings
    bool enabled;
    int max_sequence_length;
    size_t memory_pool_size;
    
    // Component configurations
    struct {
        bool enabled;
        int memory_depth;
        float decay_rate;
    } deep_memory;
    
    struct {
        bool enabled;
        int window_size;
        float omega_factor;
    } sliding_window;
    
    struct {
        bool enabled;
        float learning_rate;
        int newton_schulz_iterations;
    } muon_optimizer;
    
    struct {
        bool enabled;
        int feature_dim_multiplier;
        int polynomial_degree;
    } feature_mapping;
};

// ATLAS Memory Manager
struct atlas_memory_manager {
    struct ggml_context * main_ctx;
    ggml_gallocr_t main_allocr;
    
    // Component contexts
    struct ggml_context * deep_memory_ctx;
    struct ggml_context * omega_ctx;
    struct ggml_context * muon_ctx;
    struct ggml_context * feature_ctx;
    
    // Memory usage tracking
    size_t total_allocated;
    size_t peak_usage;
};

// ATLAS Attention Layer
struct atlas_attention_layer {
    struct atlas_config config;
    struct atlas_memory_manager memory;
    
    // Component state tensors
    struct ggml_tensor * deep_memory_state;
    struct ggml_tensor * omega_window_buffer;
    struct ggml_tensor * muon_momentum;
    struct ggml_tensor * feature_mapping_cache;
    
    // Unified attention weights
    struct ggml_tensor * attention_weights;
    struct ggml_tensor * output_projection;
    
    // Performance monitoring
    double last_forward_time;
    size_t total_operations;
    bool profiling_enabled;
};

// ATLAS Context (extends llama context)
struct atlas_context {
    struct atlas_attention_layer * layers;
    int n_layers;
    struct atlas_config config;
    bool initialized;
};

// Core ATLAS Functions
struct atlas_context * atlas_init(const struct atlas_config * config, int n_layers);
void atlas_free(struct atlas_context * atlas_ctx);

struct ggml_tensor * atlas_attention_forward(
    struct ggml_context * ctx,
    struct atlas_attention_layer * layer,
    struct ggml_tensor * input,
    struct ggml_tensor * attention_mask,
    int sequence_length,
    int head_dim
);

bool atlas_memory_init(struct atlas_memory_manager * manager, size_t pool_size);
void atlas_memory_free(struct atlas_memory_manager * manager);

// Configuration functions
struct atlas_config atlas_config_default(void);
bool atlas_config_validate(const struct atlas_config * config);

// Performance and monitoring
void atlas_enable_profiling(struct atlas_context * atlas_ctx, bool enable);
void atlas_get_performance_stats(const struct atlas_context * atlas_ctx, 
                                 double * avg_forward_time,
                                 size_t * total_ops,
                                 size_t * peak_memory);

#ifdef __cplusplus
}
#endif