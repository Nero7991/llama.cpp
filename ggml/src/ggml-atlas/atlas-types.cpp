#include "../../include/atlas/atlas-types.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Global backend registry
static atlas_backend_entry_t g_backend_registry[ATLAS_BACKEND_COUNT] = {0};

// Tensor descriptor operations
atlas_status_t atlas_tensor_desc_init(struct atlas_tensor_desc* desc,
                                      ggml_type dtype,
                                      const int64_t* shape,
                                      int n_dims) {
    if (!desc || !shape || n_dims <= 0 || n_dims > ATLAS_MAX_DIMS) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    desc->dtype = dtype;
    desc->n_dims = n_dims;
    desc->backend = ATLAS_BACKEND_CPU;
    desc->data = NULL;
    
    // Calculate strides and total size
    size_t element_size = 4; // Default to float32
    desc->data_size = element_size;
    
    for (int i = n_dims - 1; i >= 0; i--) {
        desc->shape[i] = shape[i];
        if (i == n_dims - 1) {
            desc->strides[i] = 1;
        } else {
            desc->strides[i] = desc->strides[i + 1] * shape[i + 1];
        }
        desc->data_size *= shape[i];
    }
    
    return ATLAS_STATUS_SUCCESS;
}

atlas_status_t atlas_tensor_desc_copy(struct atlas_tensor_desc* dst,
                                      const struct atlas_tensor_desc* src) {
    if (!dst || !src) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    memcpy(dst, src, sizeof(struct atlas_tensor_desc));
    
    // If source has allocated data, we need to allocate and copy
    if (src->data && src->data_size > 0) {
        dst->data = malloc(src->data_size);
        if (!dst->data) {
            return ATLAS_STATUS_OUT_OF_MEMORY;
        }
        memcpy(dst->data, src->data, src->data_size);
    }
    
    return ATLAS_STATUS_SUCCESS;
}

void atlas_tensor_desc_free(struct atlas_tensor_desc* desc) {
    if (desc && desc->data) {
        free(desc->data);
        desc->data = NULL;
        desc->data_size = 0;
    }
}

// Backend registry operations
atlas_status_t atlas_backend_register(atlas_backend_ops_t* ops, int priority) {
    if (!ops || ops->type >= ATLAS_BACKEND_COUNT) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    atlas_backend_entry_t* entry = &g_backend_registry[ops->type];
    entry->ops = ops;
    entry->priority = priority;
    
    // Check if backend is available by trying to initialize it
    void* test_context = NULL;
    if (ops->init && ops->init(&test_context) == ATLAS_STATUS_SUCCESS) {
        entry->available = true;
        if (ops->deinit) {
            ops->deinit(test_context);
        }
    } else {
        entry->available = false;
    }
    
    return ATLAS_STATUS_SUCCESS;
}

atlas_status_t atlas_backend_unregister(atlas_backend_t type) {
    if (type >= ATLAS_BACKEND_COUNT) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    atlas_backend_entry_t* entry = &g_backend_registry[type];
    entry->ops = NULL;
    entry->priority = 0;
    entry->available = false;
    
    return ATLAS_STATUS_SUCCESS;
}

atlas_backend_ops_t* atlas_backend_get(atlas_backend_t type) {
    if (type >= ATLAS_BACKEND_COUNT) {
        return NULL;
    }
    
    atlas_backend_entry_t* entry = &g_backend_registry[type];
    return entry->available ? entry->ops : NULL;
}

atlas_backend_t atlas_backend_select_best(void) {
    atlas_backend_t best = ATLAS_BACKEND_CPU;
    int best_priority = -1;
    
    for (int i = 0; i < ATLAS_BACKEND_COUNT; i++) {
        atlas_backend_entry_t* entry = &g_backend_registry[i];
        if (entry->available && entry->priority > best_priority) {
            best = (atlas_backend_t)i;
            best_priority = entry->priority;
        }
    }
    
    return best;
}

bool atlas_backend_is_available(atlas_backend_t type) {
    if (type >= ATLAS_BACKEND_COUNT) {
        return false;
    }
    return g_backend_registry[type].available;
}

// Context operations (basic implementation - full implementation would require GGML integration)
struct atlas_context* atlas_context_create(struct ggml_context* ggml_ctx,
                                          int window_size,
                                          atlas_backend_t backend) {
    struct atlas_context* ctx = (struct atlas_context*)calloc(1, sizeof(struct atlas_context));
    if (!ctx) {
        return NULL;
    }
    
    ctx->window_size = window_size > 0 ? window_size : ATLAS_DEFAULT_WINDOW_SIZE;
    ctx->current_position = 0;
    ctx->backend = backend;
    
    // Default optimization parameters
    ctx->omega_alpha = 0.01f;
    ctx->muon_iterations = 3;
    ctx->muon_momentum = 0.9f;
    ctx->kernel_type = ATLAS_KERNEL_POLYNOMIAL;
    ctx->kernel_degree = 3;
    
    // Initialize backend context
    atlas_backend_ops_t* ops = atlas_backend_get(backend);
    if (ops && ops->init) {
        if (ops->init(&ctx->backend_context) != ATLAS_STATUS_SUCCESS) {
            free(ctx);
            return NULL;
        }
    }
    
    return ctx;
}

void atlas_context_free(struct atlas_context* ctx) {
    if (!ctx) {
        return;
    }
    
    // Cleanup backend context
    atlas_backend_ops_t* ops = atlas_backend_get(ctx->backend);
    if (ops && ops->deinit && ctx->backend_context) {
        ops->deinit(ctx->backend_context);
    }
    
    free(ctx);
}

atlas_status_t atlas_context_reset(struct atlas_context* ctx) {
    if (!ctx) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    ctx->current_position = 0;
    // Reset other state as needed
    
    return ATLAS_STATUS_SUCCESS;
}

// Memory module operations (stub implementation)
atlas_status_t atlas_memory_module_init(struct atlas_memory_module* module,
                                        struct ggml_context* ctx,
                                        int input_dim, int hidden_dim, int output_dim) {
    if (!module || input_dim <= 0 || hidden_dim <= 0 || output_dim <= 0) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    module->input_dim = input_dim;
    module->hidden_dim = hidden_dim;
    module->output_dim = output_dim;
    module->activation_fn = ATLAS_ACTIVATION_GELU;
    
    // Tensor allocation would require GGML context
    // This is a placeholder for the actual implementation
    module->w1 = NULL;
    module->b1 = NULL;
    module->w2 = NULL;
    module->b2 = NULL;
    module->w_res = NULL;
    
    return ATLAS_STATUS_SUCCESS;
}

atlas_status_t atlas_memory_module_forward(struct atlas_memory_module* module,
                                           const struct ggml_tensor* input,
                                           struct ggml_tensor* output) {
    if (!module || !input || !output) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    // Forward pass implementation would require GGML operations
    // This is a placeholder for the actual implementation
    
    return ATLAS_STATUS_SUCCESS;
}