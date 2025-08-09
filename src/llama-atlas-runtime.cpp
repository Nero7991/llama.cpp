#include "atlas-runtime.h"
#include "llama-impl.h"
#include "ggml.h"
#include "llama-cpp.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Internal memory pool structure
typedef struct atlas_memory_pool {
    void * base_ptr;
    size_t total_size;
    size_t used_size;
    size_t alignment;
    pthread_mutex_t mutex;
} atlas_memory_pool;

// Internal module structure
typedef struct atlas_loaded_module {
    char name[64];
    atlas_arch_type arch_type;
    void * module_data;
    size_t memory_usage;
    bool is_active;
    float performance_gain;
    struct atlas_loaded_module * next;
} atlas_loaded_module;

// ATLAS Runtime Context Implementation
struct atlas_runtime_ctx {
    atlas_runtime_params params;
    atlas_arch_type detected_arch;
    atlas_memory_pool memory_pool;
    atlas_loaded_module * modules;
    atlas_runtime_stats stats;
    pthread_mutex_t ctx_mutex;
    bool initialized;
};

// Global error state
static __thread atlas_runtime_result g_last_error = ATLAS_RUNTIME_SUCCESS;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

// Forward declarations for architecture-specific functions
extern const atlas_arch_config * atlas_get_arch_configs(void);
extern atlas_arch_type atlas_detect_model_architecture(const struct llama_model * model);

// Memory pool implementation
static atlas_runtime_result atlas_pool_init(atlas_memory_pool * pool, size_t size) {
    if (!pool || size == 0) {
        return ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
    }

    pool->alignment = 32; // 32-byte alignment for SIMD
    pool->total_size = ((size + pool->alignment - 1) / pool->alignment) * pool->alignment;
    
    pool->base_ptr = aligned_alloc(pool->alignment, pool->total_size);
    if (!pool->base_ptr) {
        return ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION;
    }

    memset(pool->base_ptr, 0, pool->total_size);
    pool->used_size = 0;

    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool->base_ptr);
        return ATLAS_RUNTIME_ERROR_INIT_FAILED;
    }

    return ATLAS_RUNTIME_SUCCESS;
}

static void atlas_pool_cleanup(atlas_memory_pool * pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->mutex);
    if (pool->base_ptr) {
        free(pool->base_ptr);
        pool->base_ptr = NULL;
    }
    pool->total_size = 0;
    pool->used_size = 0;
    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);
}

static void * atlas_pool_alloc(atlas_memory_pool * pool, size_t size) {
    if (!pool || size == 0) return NULL;

    pthread_mutex_lock(&pool->mutex);
    
    size_t aligned_size = ((size + pool->alignment - 1) / pool->alignment) * pool->alignment;
    
    if (pool->used_size + aligned_size > pool->total_size) {
        pthread_mutex_unlock(&pool->mutex);
        return NULL;
    }

    void * ptr = (char*)pool->base_ptr + pool->used_size;
    pool->used_size += aligned_size;
    
    pthread_mutex_unlock(&pool->mutex);
    return ptr;
}

// Core runtime functions implementation
atlas_runtime_params atlas_runtime_default_params(void) {
    atlas_runtime_params params;
    params.enable_atlas = true;
    params.auto_detect_arch = true;
    params.sparsity_threshold = 0.1f;
    params.max_modules = 16;
    params.memory_pool_size = 256 * 1024 * 1024; // 256MB default
    params.arch_override = NULL;
    params.enable_lazy_loading = true;
    params.optimization_level = 1.0f;
    return params;
}

bool atlas_runtime_validate_params(const atlas_runtime_params * params) {
    if (!params) return false;
    if (params->sparsity_threshold < 0.0f || params->sparsity_threshold > 1.0f) return false;
    if (params->max_modules < 1 || params->max_modules > 1024) return false;
    if (params->memory_pool_size < 1024 * 1024) return false; // Min 1MB
    if (params->optimization_level < 0.0f || params->optimization_level > 2.0f) return false;
    return true;
}

atlas_runtime_result atlas_runtime_init(
    atlas_runtime_ctx ** ctx,
    const struct llama_model * model,
    const atlas_runtime_params * params
) {
    if (!ctx || !model) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    atlas_runtime_params default_params = atlas_runtime_default_params();
    if (!params) params = &default_params;

    if (!atlas_runtime_validate_params(params)) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&g_init_mutex);

    atlas_runtime_ctx * new_ctx = (atlas_runtime_ctx*)malloc(sizeof(atlas_runtime_ctx));
    if (!new_ctx) {
        pthread_mutex_unlock(&g_init_mutex);
        g_last_error = ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION;
        return g_last_error;
    }

    memset(new_ctx, 0, sizeof(atlas_runtime_ctx));
    new_ctx->params = *params;

    // Initialize context mutex
    if (pthread_mutex_init(&new_ctx->ctx_mutex, NULL) != 0) {
        free(new_ctx);
        pthread_mutex_unlock(&g_init_mutex);
        g_last_error = ATLAS_RUNTIME_ERROR_INIT_FAILED;
        return g_last_error;
    }

    // Detect architecture
    if (params->auto_detect_arch) {
        new_ctx->detected_arch = atlas_detect_model_architecture(model);
    } else if (params->arch_override) {
        // Parse architecture override
        if (strcmp(params->arch_override, "llama") == 0) {
            new_ctx->detected_arch = ATLAS_ARCH_LLAMA;
        } else if (strcmp(params->arch_override, "mistral") == 0) {
            new_ctx->detected_arch = ATLAS_ARCH_MISTRAL;
        } else if (strcmp(params->arch_override, "phi") == 0) {
            new_ctx->detected_arch = ATLAS_ARCH_PHI;
        } else if (strcmp(params->arch_override, "gemma") == 0) {
            new_ctx->detected_arch = ATLAS_ARCH_GEMMA;
        } else {
            new_ctx->detected_arch = ATLAS_ARCH_UNKNOWN;
        }
    } else {
        new_ctx->detected_arch = ATLAS_ARCH_UNKNOWN;
    }

    if (new_ctx->detected_arch == ATLAS_ARCH_UNKNOWN) {
        pthread_mutex_destroy(&new_ctx->ctx_mutex);
        free(new_ctx);
        pthread_mutex_unlock(&g_init_mutex);
        g_last_error = ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED;
        return g_last_error;
    }

    // Initialize memory pool
    atlas_runtime_result pool_result = atlas_pool_init(&new_ctx->memory_pool, params->memory_pool_size);
    if (pool_result != ATLAS_RUNTIME_SUCCESS) {
        pthread_mutex_destroy(&new_ctx->ctx_mutex);
        free(new_ctx);
        pthread_mutex_unlock(&g_init_mutex);
        g_last_error = pool_result;
        return g_last_error;
    }

    // Initialize stats
    memset(&new_ctx->stats, 0, sizeof(atlas_runtime_stats));
    new_ctx->initialized = true;

    *ctx = new_ctx;
    pthread_mutex_unlock(&g_init_mutex);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

atlas_runtime_result atlas_runtime_cleanup(atlas_runtime_ctx * ctx) {
    if (!ctx) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&ctx->ctx_mutex);

    // Cleanup modules
    atlas_loaded_module * current = ctx->modules;
    while (current) {
        atlas_loaded_module * next = current->next;
        if (current->module_data) {
            free(current->module_data);
        }
        free(current);
        current = next;
    }
    ctx->modules = NULL;

    // Cleanup memory pool
    atlas_pool_cleanup(&ctx->memory_pool);

    ctx->initialized = false;
    pthread_mutex_unlock(&ctx->ctx_mutex);
    pthread_mutex_destroy(&ctx->ctx_mutex);

    free(ctx);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

atlas_runtime_result atlas_runtime_enable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
) {
    if (!runtime_ctx || !llama_ctx) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&runtime_ctx->ctx_mutex);

    // Enable ATLAS optimizations for this context
    // This would integrate with llama.cpp's internal context structure
    // For now, we just mark it as enabled

    pthread_mutex_unlock(&runtime_ctx->ctx_mutex);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

atlas_runtime_result atlas_runtime_disable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
) {
    if (!runtime_ctx || !llama_ctx) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&runtime_ctx->ctx_mutex);

    // Disable ATLAS optimizations for this context
    // This would integrate with llama.cpp's internal context structure

    pthread_mutex_unlock(&runtime_ctx->ctx_mutex);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

const char * atlas_runtime_error_string(atlas_runtime_result error) {
    switch (error) {
        case ATLAS_RUNTIME_SUCCESS:
            return "Success";
        case ATLAS_RUNTIME_ERROR_INVALID_PARAMS:
            return "Invalid parameters";
        case ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED:
            return "Architecture not supported";
        case ATLAS_RUNTIME_ERROR_INIT_FAILED:
            return "Initialization failed";
        case ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND:
            return "Module not found";
        case ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED:
            return "Memory pool exhausted";
        case ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL:
            return "Incompatible model";
        default:
            return "Unknown error";
    }
}

atlas_runtime_result atlas_runtime_get_last_error(void) {
    return g_last_error;
}

size_t atlas_runtime_get_pool_usage(atlas_runtime_ctx * ctx) {
    if (!ctx) return 0;
    
    pthread_mutex_lock(&ctx->memory_pool.mutex);
    size_t usage = ctx->memory_pool.used_size;
    pthread_mutex_unlock(&ctx->memory_pool.mutex);
    
    return usage;
}

size_t atlas_runtime_get_pool_capacity(atlas_runtime_ctx * ctx) {
    if (!ctx) return 0;
    return ctx->memory_pool.total_size;
}

atlas_runtime_result atlas_runtime_get_stats(
    atlas_runtime_ctx * ctx,
    atlas_runtime_stats * stats
) {
    if (!ctx || !stats) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&ctx->ctx_mutex);
    *stats = ctx->stats;
    pthread_mutex_unlock(&ctx->ctx_mutex);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

atlas_runtime_result atlas_runtime_reset_stats(atlas_runtime_ctx * ctx) {
    if (!ctx) {
        g_last_error = ATLAS_RUNTIME_ERROR_INVALID_PARAMS;
        return g_last_error;
    }

    pthread_mutex_lock(&ctx->ctx_mutex);
    memset(&ctx->stats, 0, sizeof(atlas_runtime_stats));
    pthread_mutex_unlock(&ctx->ctx_mutex);

    g_last_error = ATLAS_RUNTIME_SUCCESS;
    return ATLAS_RUNTIME_SUCCESS;
}

#ifdef __cplusplus
}
#endif