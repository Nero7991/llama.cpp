#pragma once

#include "llama.h"
#include "ggml.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS Runtime Error Codes
typedef enum {
    ATLAS_RUNTIME_SUCCESS = 0,
    ATLAS_RUNTIME_ERROR_INVALID_PARAMS,
    ATLAS_RUNTIME_ERROR_MEMORY_ALLOCATION,
    ATLAS_RUNTIME_ERROR_ARCH_UNSUPPORTED,
    ATLAS_RUNTIME_ERROR_INIT_FAILED,
    ATLAS_RUNTIME_ERROR_MODULE_NOT_FOUND,
    ATLAS_RUNTIME_ERROR_POOL_EXHAUSTED,
    ATLAS_RUNTIME_ERROR_INCOMPATIBLE_MODEL
} atlas_runtime_result;

// ATLAS Architecture Types
typedef enum {
    ATLAS_ARCH_AUTO = 0,
    ATLAS_ARCH_LLAMA,
    ATLAS_ARCH_MISTRAL,
    ATLAS_ARCH_PHI,
    ATLAS_ARCH_GEMMA,
    ATLAS_ARCH_UNKNOWN
} atlas_arch_type;

// ATLAS Runtime Parameters
typedef struct atlas_runtime_params {
    bool enable_atlas;
    bool auto_detect_arch;
    float sparsity_threshold;
    int max_modules;
    size_t memory_pool_size;
    const char * arch_override;
    bool enable_lazy_loading;
    float optimization_level;
} atlas_runtime_params;

// ATLAS Runtime Context (opaque)
typedef struct atlas_runtime_ctx atlas_runtime_ctx;

// ATLAS Architecture Configuration
typedef struct atlas_arch_config {
    atlas_arch_type type;
    const char * name;
    float sparsity_default;
    int layers_optimal;
    size_t memory_requirements;
    bool supports_dynamic_loading;
} atlas_arch_config;

// ATLAS Module Information
typedef struct atlas_module_info {
    const char * name;
    atlas_arch_type arch_type;
    size_t memory_usage;
    bool is_active;
    float performance_gain;
} atlas_module_info;

// Core Runtime Functions
atlas_runtime_result atlas_runtime_init(
    atlas_runtime_ctx ** ctx,
    const struct llama_model * model,
    const atlas_runtime_params * params
);

atlas_runtime_result atlas_runtime_cleanup(atlas_runtime_ctx * ctx);

atlas_runtime_result atlas_runtime_enable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
);

atlas_runtime_result atlas_runtime_disable_for_context(
    atlas_runtime_ctx * runtime_ctx,
    struct llama_context * llama_ctx
);

// Architecture Detection
atlas_arch_type atlas_runtime_detect_architecture(const struct llama_model * model);
const atlas_arch_config * atlas_runtime_get_arch_config(atlas_arch_type arch);
const char * atlas_runtime_arch_name(atlas_arch_type arch);

// Module Management
atlas_runtime_result atlas_runtime_load_module(
    atlas_runtime_ctx * ctx,
    const char * module_name,
    const atlas_arch_config * config
);

atlas_runtime_result atlas_runtime_unload_module(
    atlas_runtime_ctx * ctx,
    const char * module_name
);

int atlas_runtime_get_active_modules(
    atlas_runtime_ctx * ctx,
    atlas_module_info * modules,
    int max_modules
);

// Memory Pool Management
atlas_runtime_result atlas_runtime_resize_pool(
    atlas_runtime_ctx * ctx,
    size_t new_size
);

size_t atlas_runtime_get_pool_usage(atlas_runtime_ctx * ctx);
size_t atlas_runtime_get_pool_capacity(atlas_runtime_ctx * ctx);

// Parameter Management
atlas_runtime_params atlas_runtime_default_params(void);
bool atlas_runtime_validate_params(const atlas_runtime_params * params);

// Error Handling
const char * atlas_runtime_error_string(atlas_runtime_result error);
atlas_runtime_result atlas_runtime_get_last_error(void);

// Performance Metrics
typedef struct atlas_runtime_stats {
    size_t total_memory_used;
    size_t modules_loaded;
    float average_speedup;
    uint64_t operations_processed;
} atlas_runtime_stats;

atlas_runtime_result atlas_runtime_get_stats(
    atlas_runtime_ctx * ctx,
    atlas_runtime_stats * stats
);

atlas_runtime_result atlas_runtime_reset_stats(atlas_runtime_ctx * ctx);

#ifdef __cplusplus
}
#endif