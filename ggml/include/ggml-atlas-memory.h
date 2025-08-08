#ifndef GGML_ATLAS_MEMORY_H
#define GGML_ATLAS_MEMORY_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS activation functions
typedef enum {
    GGML_ATLAS_ACT_GELU,
    GGML_ATLAS_ACT_RELU,
    GGML_ATLAS_ACT_SILU,
} ggml_atlas_activation_t;

// ATLAS memory configuration
struct ggml_atlas_memory_config {
    int32_t input_dim;
    int32_t hidden_dim;
    int32_t output_dim;
    ggml_atlas_activation_t activation;
    float dropout_rate;
    bool use_residual;
};

// ATLAS memory context
struct ggml_atlas_memory_context {
    struct ggml_atlas_memory_config config;
    struct ggml_tensor * w1;      // input -> hidden weights
    struct ggml_tensor * b1;      // hidden biases
    struct ggml_tensor * w2;      // hidden -> output weights
    struct ggml_tensor * b2;      // output biases
    struct ggml_context * ctx;
    size_t memory_size;
};

// Memory pool for ATLAS operations
struct ggml_atlas_memory_pool {
    void * data;
    size_t size;
    size_t used;
    size_t max_vectors;
};

// API functions
GGML_API struct ggml_atlas_memory_context * ggml_atlas_memory_init(
    const struct ggml_atlas_memory_config * config);

GGML_API void ggml_atlas_memory_free(struct ggml_atlas_memory_context * ctx);

GGML_API struct ggml_tensor * ggml_atlas_memory_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input);

GGML_API struct ggml_atlas_memory_pool * ggml_atlas_memory_pool_init(size_t max_vectors, size_t vector_size);

GGML_API void ggml_atlas_memory_pool_free(struct ggml_atlas_memory_pool * pool);

GGML_API void * ggml_atlas_memory_pool_alloc(struct ggml_atlas_memory_pool * pool, size_t size);

GGML_API void ggml_atlas_memory_pool_reset(struct ggml_atlas_memory_pool * pool);

// Backend-specific functions
GGML_API bool ggml_atlas_memory_supported(void);
GGML_API void ggml_atlas_memory_set_backend(void * backend);

// Performance monitoring
struct ggml_atlas_memory_stats {
    uint64_t total_ops;
    uint64_t total_time_us;
    size_t memory_used;
    size_t memory_peak;
};

GGML_API void ggml_atlas_memory_get_stats(struct ggml_atlas_memory_stats * stats);
GGML_API void ggml_atlas_memory_reset_stats(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_ATLAS_MEMORY_H