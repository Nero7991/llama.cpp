#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "../ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations  
struct ggml_tensor;
struct ggml_context;

// ATLAS configuration constants
#define ATLAS_MAX_DIMS 4
#define ATLAS_MAX_BACKENDS 8
#define ATLAS_DEFAULT_WINDOW_SIZE 512
#define ATLAS_DEFAULT_MEMORY_POOL_SIZE (256 * 1024 * 1024) // 256MB
#define ATLAS_DEFAULT_ALIGNMENT 32

// ATLAS status codes
typedef enum {
    ATLAS_STATUS_SUCCESS = 0,
    ATLAS_STATUS_ERROR = -1,
    ATLAS_STATUS_OUT_OF_MEMORY = -2,
    ATLAS_STATUS_INVALID_ARGUMENT = -3,
    ATLAS_STATUS_NOT_SUPPORTED = -4,
    ATLAS_STATUS_BACKEND_ERROR = -5,
} atlas_status_t;

// ATLAS backend types
typedef enum {
    ATLAS_BACKEND_CPU = 0,
    ATLAS_BACKEND_CUDA = 1,
    ATLAS_BACKEND_METAL = 2,
    ATLAS_BACKEND_OPENCL = 3,
    ATLAS_BACKEND_COUNT
} atlas_backend_t;

// ATLAS activation functions
typedef enum {
    ATLAS_ACTIVATION_NONE = 0,
    ATLAS_ACTIVATION_RELU,
    ATLAS_ACTIVATION_GELU,
    ATLAS_ACTIVATION_SILU,
    ATLAS_ACTIVATION_TANH,
} atlas_activation_t;

// ATLAS kernel types for feature mapping
typedef enum {
    ATLAS_KERNEL_LINEAR = 0,
    ATLAS_KERNEL_POLYNOMIAL,
    ATLAS_KERNEL_EXPONENTIAL,
    ATLAS_KERNEL_RBF,
} atlas_kernel_type_t;

// ATLAS memory allocation strategy
typedef enum {
    ATLAS_ALLOC_FIRST_FIT = 0,
    ATLAS_ALLOC_BEST_FIT,
    ATLAS_ALLOC_WORST_FIT,
} atlas_alloc_strategy_t;

// ATLAS tensor descriptor
struct atlas_tensor_desc {
    ggml_type dtype;
    int64_t shape[ATLAS_MAX_DIMS];
    int64_t strides[ATLAS_MAX_DIMS];
    int n_dims;
    size_t data_size;
    void* data;
    atlas_backend_t backend;
};

// ATLAS memory pool flags
typedef uint32_t atlas_memory_pool_flags_t;
#define ATLAS_MEMORY_POOL_FLAG_NONE 0x00
#define ATLAS_MEMORY_POOL_FLAG_GROWABLE 0x01
#define ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE 0x02
#define ATLAS_MEMORY_POOL_FLAG_ZERO_INIT 0x04

// ATLAS memory pool
typedef struct atlas_memory_pool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    size_t alignment;
    atlas_memory_pool_flags_t flags;
    atlas_alloc_strategy_t allocation_strategy;
    void* allocator_state; // Backend-specific state
} atlas_memory_pool_t;

// ATLAS memory module (2-layer MLP with residual)
struct atlas_memory_module {
    // Layer 1: Input -> Hidden
    struct ggml_tensor* w1;     // [input_dim, hidden_dim]
    struct ggml_tensor* b1;     // [hidden_dim]
    
    // Layer 2: Hidden -> Output
    struct ggml_tensor* w2;     // [hidden_dim, output_dim]
    struct ggml_tensor* b2;     // [output_dim]
    
    // Residual connection
    struct ggml_tensor* w_res;  // [input_dim, output_dim]
    
    // Configuration
    int input_dim;
    int hidden_dim;
    int output_dim;
    atlas_activation_t activation_fn;
};

// ATLAS context for managing the memory optimization
struct atlas_context {
    // Memory components
    struct atlas_memory_module memory;
    struct ggml_tensor* momentum_state;    // Muon optimizer state
    
    // Sliding window management
    struct ggml_tensor* context_window;    // Circular buffer
    int window_size;                       // W in paper
    int current_position;                  // Current write head
    
    // Optimization parameters
    float omega_alpha;                     // Learning rate
    int muon_iterations;                   // Newton-Schulz steps
    float muon_momentum;                   // Momentum coefficient
    
    // Feature mapping
    atlas_kernel_type_t kernel_type;       // Polynomial/Exponential
    int kernel_degree;                     // For polynomial kernels
    
    // Backend
    atlas_backend_t backend;
    void* backend_context;                 // Backend-specific context
};

// ATLAS backend operations interface
typedef struct atlas_backend_ops {
    const char* name;
    atlas_backend_t type;
    
    // Initialization and cleanup
    atlas_status_t (*init)(void** context);
    atlas_status_t (*deinit)(void* context);
    
    // Memory operations
    atlas_status_t (*alloc)(void* context, size_t size, void** ptr);
    atlas_status_t (*free)(void* context, void* ptr);
    atlas_status_t (*copy)(void* context, void* dst, const void* src, size_t size);
    
    // Computation operations
    atlas_status_t (*gemm)(void* context, const struct atlas_tensor_desc* a,
                          const struct atlas_tensor_desc* b,
                          struct atlas_tensor_desc* c);
    atlas_status_t (*activation)(void* context, atlas_activation_t type,
                                 struct atlas_tensor_desc* tensor);
    
    // Synchronization
    atlas_status_t (*sync)(void* context);
} atlas_backend_ops_t;

// ATLAS backend registry entry
typedef struct atlas_backend_entry {
    atlas_backend_ops_t* ops;
    int priority;
    bool available;
} atlas_backend_entry_t;

// Function declarations for ATLAS operations

// Memory pool operations
atlas_memory_pool_t* atlas_memory_pool_create(size_t initial_size, atlas_memory_pool_flags_t flags);
void atlas_memory_pool_destroy(atlas_memory_pool_t* pool);
atlas_status_t atlas_memory_pool_reset(atlas_memory_pool_t* pool);
atlas_status_t atlas_memory_alloc(atlas_memory_pool_t* pool, size_t size, size_t alignment, void** ptr);
atlas_status_t atlas_memory_free(atlas_memory_pool_t* pool, void* ptr);
size_t atlas_memory_pool_get_used(const atlas_memory_pool_t* pool);
size_t atlas_memory_pool_get_available(const atlas_memory_pool_t* pool);

// Backend operations
atlas_status_t atlas_backend_register(atlas_backend_ops_t* ops, int priority);
atlas_status_t atlas_backend_unregister(atlas_backend_t type);
atlas_backend_ops_t* atlas_backend_get(atlas_backend_t type);
atlas_backend_t atlas_backend_select_best(void);
bool atlas_backend_is_available(atlas_backend_t type);

// Context operations
struct atlas_context* atlas_context_create(struct ggml_context* ggml_ctx,
                                          int window_size,
                                          atlas_backend_t backend);
void atlas_context_free(struct atlas_context* ctx);
atlas_status_t atlas_context_reset(struct atlas_context* ctx);

// Memory module operations
atlas_status_t atlas_memory_module_init(struct atlas_memory_module* module,
                                        struct ggml_context* ctx,
                                        int input_dim, int hidden_dim, int output_dim);
atlas_status_t atlas_memory_module_forward(struct atlas_memory_module* module,
                                           const struct ggml_tensor* input,
                                           struct ggml_tensor* output);

// Tensor descriptor operations
atlas_status_t atlas_tensor_desc_init(struct atlas_tensor_desc* desc,
                                      ggml_type dtype,
                                      const int64_t* shape,
                                      int n_dims);
atlas_status_t atlas_tensor_desc_copy(struct atlas_tensor_desc* dst,
                                      const struct atlas_tensor_desc* src);
void atlas_tensor_desc_free(struct atlas_tensor_desc* desc);

// Additional backend functions
void atlas_backend_init_all(void);
const char* atlas_backend_name(atlas_backend_t type);
atlas_backend_t atlas_backend_auto_select(void);
atlas_status_t atlas_newton_schulz_iteration(float* matrix, float* inverse, 
                                            int dim, int iterations);

// Memory pool statistics
size_t atlas_memory_pool_get_num_allocations(const atlas_memory_pool_t* pool);
size_t atlas_memory_pool_get_peak_usage(const atlas_memory_pool_t* pool);

#ifdef __cplusplus
}
#endif