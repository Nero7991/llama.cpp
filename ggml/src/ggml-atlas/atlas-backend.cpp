#include "../../include/atlas/atlas-types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// CPU Backend Implementation
static atlas_status_t cpu_backend_init(void** context) {
    *context = calloc(1, sizeof(int)); // Minimal context
    if (!*context) {
        return ATLAS_STATUS_OUT_OF_MEMORY;
    }
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_deinit(void* context) {
    if (context) {
        free(context);
    }
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_alloc(void* context, size_t size, void** ptr) {
    (void)context; // Unused
    *ptr = aligned_alloc(ATLAS_DEFAULT_ALIGNMENT, size);
    if (!*ptr) {
        return ATLAS_STATUS_OUT_OF_MEMORY;
    }
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_free(void* context, void* ptr) {
    (void)context; // Unused
    free(ptr);
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_copy(void* context, void* dst, const void* src, size_t size) {
    (void)context; // Unused
    memcpy(dst, src, size);
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_gemm(void* context, 
                                       const struct atlas_tensor_desc* a,
                                       const struct atlas_tensor_desc* b,
                                       struct atlas_tensor_desc* c) {
    (void)context; // Unused
    
    if (!a || !b || !c || !a->data || !b->data || !c->data) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    // Simple GEMM implementation for CPU
    // Assumes 2D matrices with compatible dimensions
    if (a->n_dims != 2 || b->n_dims != 2 || c->n_dims != 2) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];
    
    if (b->shape[0] != k || c->shape[0] != m || c->shape[1] != n) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* c_data = (float*)c->data;
    
    // Simple matrix multiplication
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; l++) {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
    
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_activation(void* context,
                                            atlas_activation_t type,
                                            struct atlas_tensor_desc* tensor) {
    (void)context; // Unused
    
    if (!tensor || !tensor->data) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    size_t num_elements = tensor->data_size / sizeof(float);
    float* data = (float*)tensor->data;
    
    switch (type) {
        case ATLAS_ACTIVATION_RELU:
            for (size_t i = 0; i < num_elements; i++) {
                if (data[i] < 0) data[i] = 0;
            }
            break;
            
        case ATLAS_ACTIVATION_GELU:
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            for (size_t i = 0; i < num_elements; i++) {
                float x = data[i];
                float x3 = x * x * x;
                float arg = 0.7978845608f * (x + 0.044715f * x3);
                float tanh_val = tanhf(arg);
                data[i] = 0.5f * x * (1.0f + tanh_val);
            }
            break;
            
        case ATLAS_ACTIVATION_SILU:
            // SiLU/Swish: x * sigmoid(x)
            for (size_t i = 0; i < num_elements; i++) {
                float x = data[i];
                data[i] = x / (1.0f + expf(-x));
            }
            break;
            
        case ATLAS_ACTIVATION_TANH:
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = tanhf(data[i]);
            }
            break;
            
        case ATLAS_ACTIVATION_NONE:
        default:
            // No activation
            break;
    }
    
    return ATLAS_STATUS_SUCCESS;
}

static atlas_status_t cpu_backend_sync(void* context) {
    (void)context; // Unused - CPU backend doesn't need synchronization
    return ATLAS_STATUS_SUCCESS;
}

// CPU backend operations structure
static atlas_backend_ops_t cpu_backend_ops = {
    "CPU",                    // name
    ATLAS_BACKEND_CPU,        // type
    cpu_backend_init,         // init
    cpu_backend_deinit,       // deinit
    cpu_backend_alloc,        // alloc
    cpu_backend_free,         // free
    cpu_backend_copy,         // copy
    cpu_backend_gemm,         // gemm
    cpu_backend_activation,   // activation
    cpu_backend_sync,         // sync
};

// CUDA backend stubs (would be implemented in atlas-cuda.cpp when CUDA is available)
#ifdef GGML_USE_CUDA
static atlas_backend_ops_t cuda_backend_ops = {
    .name = "CUDA",
    .type = ATLAS_BACKEND_CUDA,
    .init = NULL,    // Would be implemented in atlas-cuda.cpp
    .deinit = NULL,
    .alloc = NULL,
    .free = NULL,
    .copy = NULL,
    .gemm = NULL,
    .activation = NULL,
    .sync = NULL,
};
#endif

// Backend initialization function
void atlas_backend_init_all(void) {
    // Register CPU backend with priority 1
    atlas_backend_register(&cpu_backend_ops, 1);
    
#ifdef GGML_USE_CUDA
    // Register CUDA backend with priority 10 (higher priority)
    atlas_backend_register(&cuda_backend_ops, 10);
#endif
}

// Helper function to get backend name
const char* atlas_backend_name(atlas_backend_t type) {
    atlas_backend_ops_t* ops = atlas_backend_get(type);
    return ops ? ops->name : "Unknown";
}

// Helper function to perform backend selection based on availability
atlas_backend_t atlas_backend_auto_select(void) {
    // Try backends in order of preference
#ifdef GGML_USE_CUDA
    if (atlas_backend_is_available(ATLAS_BACKEND_CUDA)) {
        return ATLAS_BACKEND_CUDA;
    }
#endif
    
    // Fallback to CPU
    return ATLAS_BACKEND_CPU;
}

// Math helper for CPU backend
#include <math.h>

// Newton-Schulz iteration for matrix inverse approximation (used by Muon optimizer)
atlas_status_t atlas_newton_schulz_iteration(float* matrix, float* inverse, 
                                            int dim, int iterations) {
    if (!matrix || !inverse || dim <= 0 || iterations <= 0) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    // Initialize inverse as scaled identity
    float scale = 2.0f / (dim * dim);
    for (int i = 0; i < dim * dim; i++) {
        inverse[i] = (i % (dim + 1) == 0) ? scale : 0.0f;
    }
    
    // Temporary matrices for computation
    float* temp1 = (float*)calloc(dim * dim, sizeof(float));
    float* temp2 = (float*)calloc(dim * dim, sizeof(float));
    if (!temp1 || !temp2) {
        free(temp1);
        free(temp2);
        return ATLAS_STATUS_OUT_OF_MEMORY;
    }
    
    // Newton-Schulz iterations: X_{n+1} = X_n * (2I - A * X_n)
    for (int iter = 0; iter < iterations; iter++) {
        // temp1 = A * X_n
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < dim; k++) {
                    sum += matrix[i * dim + k] * inverse[k * dim + j];
                }
                temp1[i * dim + j] = sum;
            }
        }
        
        // temp2 = 2I - temp1
        for (int i = 0; i < dim * dim; i++) {
            temp2[i] = -temp1[i];
            if (i % (dim + 1) == 0) {
                temp2[i] += 2.0f;
            }
        }
        
        // inverse = X_n * temp2
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < dim; k++) {
                    sum += inverse[i * dim + k] * temp2[k * dim + j];
                }
                temp1[i * dim + j] = sum;
            }
        }
        
        // Copy result back
        memcpy(inverse, temp1, dim * dim * sizeof(float));
    }
    
    free(temp1);
    free(temp2);
    
    return ATLAS_STATUS_SUCCESS;
}