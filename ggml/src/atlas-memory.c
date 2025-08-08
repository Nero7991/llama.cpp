#include "ggml-atlas-memory.h"
#include "ggml-impl.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

// Global statistics
static struct ggml_atlas_memory_stats g_atlas_stats = {0};
static void * g_atlas_backend = NULL;

// Forward declarations
extern bool ggml_atlas_memory_cpu_supported(void);
extern struct ggml_tensor * ggml_atlas_memory_cpu_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input);

#ifdef GGML_USE_CUDA
extern bool ggml_atlas_memory_cuda_supported(void);
extern struct ggml_tensor * ggml_atlas_memory_cuda_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input);
#endif

struct ggml_atlas_memory_context * ggml_atlas_memory_init(
    const struct ggml_atlas_memory_config * config) {
    
    assert(config != NULL);
    assert(config->input_dim > 0);
    assert(config->hidden_dim > 0);
    assert(config->output_dim > 0);
    assert(config->dropout_rate >= 0.0f && config->dropout_rate < 1.0f);

    struct ggml_atlas_memory_context * atlas_ctx = calloc(1, sizeof(struct ggml_atlas_memory_context));
    if (!atlas_ctx) {
        return NULL;
    }

    atlas_ctx->config = *config;

    // Calculate memory requirements with extra space for GGML overhead
    size_t w1_size = config->input_dim * config->hidden_dim * sizeof(float);
    size_t b1_size = config->hidden_dim * sizeof(float);
    size_t w2_size = config->hidden_dim * config->output_dim * sizeof(float);
    size_t b2_size = config->output_dim * sizeof(float);
    
    // Add significant overhead for GGML tensor metadata and alignment
    atlas_ctx->memory_size = (w1_size + b1_size + w2_size + b2_size) * 2 + 64 * 1024; // 64KB extra

    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size = atlas_ctx->memory_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    atlas_ctx->ctx = ggml_init(params);
    if (!atlas_ctx->ctx) {
        free(atlas_ctx);
        return NULL;
    }

    // Create weight tensors
    atlas_ctx->w1 = ggml_new_tensor_2d(atlas_ctx->ctx, GGML_TYPE_F32, config->input_dim, config->hidden_dim);
    atlas_ctx->b1 = ggml_new_tensor_1d(atlas_ctx->ctx, GGML_TYPE_F32, config->hidden_dim);
    atlas_ctx->w2 = ggml_new_tensor_2d(atlas_ctx->ctx, GGML_TYPE_F32, config->hidden_dim, config->output_dim);
    atlas_ctx->b2 = ggml_new_tensor_1d(atlas_ctx->ctx, GGML_TYPE_F32, config->output_dim);

    if (!atlas_ctx->w1 || !atlas_ctx->b1 || !atlas_ctx->w2 || !atlas_ctx->b2) {
        ggml_free(atlas_ctx->ctx);
        free(atlas_ctx);
        return NULL;
    }

    // Initialize weights with Xavier/Glorot initialization
    float w1_scale = sqrtf(2.0f / (config->input_dim + config->hidden_dim));
    float w2_scale = sqrtf(2.0f / (config->hidden_dim + config->output_dim));

    // Initialize w1
    float * w1_data = (float *)atlas_ctx->w1->data;
    for (int i = 0; i < config->input_dim * config->hidden_dim; i++) {
        w1_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * w1_scale;
    }

    // Initialize b1 to zero
    memset(atlas_ctx->b1->data, 0, config->hidden_dim * sizeof(float));

    // Initialize w2
    float * w2_data = (float *)atlas_ctx->w2->data;
    for (int i = 0; i < config->hidden_dim * config->output_dim; i++) {
        w2_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * w2_scale;
    }

    // Initialize b2 to zero
    memset(atlas_ctx->b2->data, 0, config->output_dim * sizeof(float));

    return atlas_ctx;
}

void ggml_atlas_memory_free(struct ggml_atlas_memory_context * ctx) {
    if (!ctx) return;
    
    if (ctx->ctx) {
        ggml_free(ctx->ctx);
    }
    free(ctx);
}

struct ggml_tensor * ggml_atlas_memory_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input) {
    
    assert(ctx != NULL);
    assert(atlas_ctx != NULL);
    assert(input != NULL);
    
    int64_t start_time = 0; // Placeholder for timing

    struct ggml_tensor * result = NULL;

    // Try CUDA first if available
#ifdef GGML_USE_CUDA
    if (g_atlas_backend && ggml_atlas_memory_cuda_supported()) {
        result = ggml_atlas_memory_cuda_forward(ctx, atlas_ctx, input);
    }
#endif

    // Fallback to CPU
    if (!result && ggml_atlas_memory_cpu_supported()) {
        result = ggml_atlas_memory_cpu_forward(ctx, atlas_ctx, input);
    }

    // Update statistics
    g_atlas_stats.total_ops++;

    return result;
}

struct ggml_atlas_memory_pool * ggml_atlas_memory_pool_init(size_t max_vectors, size_t vector_size) {
    struct ggml_atlas_memory_pool * pool = calloc(1, sizeof(struct ggml_atlas_memory_pool));
    if (!pool) return NULL;

    pool->size = max_vectors * vector_size;
    pool->max_vectors = max_vectors;
    pool->used = 0;

    pool->data = malloc(pool->size);
    if (!pool->data) {
        free(pool);
        return NULL;
    }

    return pool;
}

void ggml_atlas_memory_pool_free(struct ggml_atlas_memory_pool * pool) {
    if (!pool) return;
    
    if (pool->data) {
        free(pool->data);
    }
    free(pool);
}

void * ggml_atlas_memory_pool_alloc(struct ggml_atlas_memory_pool * pool, size_t size) {
    if (!pool || pool->used + size > pool->size) {
        return NULL;
    }

    void * ptr = (char *)pool->data + pool->used;
    pool->used += size;
    
    g_atlas_stats.memory_used = pool->used;
    if (pool->used > g_atlas_stats.memory_peak) {
        g_atlas_stats.memory_peak = pool->used;
    }

    return ptr;
}

void ggml_atlas_memory_pool_reset(struct ggml_atlas_memory_pool * pool) {
    if (pool) {
        pool->used = 0;
    }
}

bool ggml_atlas_memory_supported(void) {
#ifdef GGML_USE_CUDA
    if (ggml_atlas_memory_cuda_supported()) {
        return true;
    }
#endif
    return ggml_atlas_memory_cpu_supported();
}

void ggml_atlas_memory_set_backend(void * backend) {
    g_atlas_backend = backend;
}

void ggml_atlas_memory_get_stats(struct ggml_atlas_memory_stats * stats) {
    if (stats) {
        *stats = g_atlas_stats;
    }
}

void ggml_atlas_memory_reset_stats(void) {
    memset(&g_atlas_stats, 0, sizeof(g_atlas_stats));
}