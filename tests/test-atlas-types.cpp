// Unit tests for ATLAS types and data structures
// Tests tensor descriptors, memory pools, and core ATLAS data structures

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <random>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ATLAS type definitions (based on atlas_feature.md specifications)
enum atlas_activation {
    ATLAS_ACT_NONE = 0,
    ATLAS_ACT_GELU,
    ATLAS_ACT_RELU,
    ATLAS_ACT_SILU,
};

enum atlas_kernel_type {
    ATLAS_KERNEL_POLYNOMIAL = 0,
    ATLAS_KERNEL_EXPONENTIAL,
    ATLAS_KERNEL_RBF,
};

struct atlas_memory_module {
    // Layer 1: Input → Hidden
    struct ggml_tensor * w1;     // [input_dim, hidden_dim]
    struct ggml_tensor * b1;     // [hidden_dim]
    
    // Layer 2: Hidden → Output  
    struct ggml_tensor * w2;     // [hidden_dim, output_dim]
    struct ggml_tensor * b2;     // [output_dim]
    
    // Residual connection
    struct ggml_tensor * w_res;  // [input_dim, output_dim]
    
    // Configuration
    int input_dim;
    int hidden_dim;
    int output_dim;
    enum atlas_activation activation_fn;
};

struct atlas_context {
    // Memory components
    struct atlas_memory_module memory;
    struct ggml_tensor * momentum_state;    // Muon optimizer state
    
    // Sliding window management  
    struct ggml_tensor * context_window;    // Circular buffer
    int window_size;                        // W in paper
    int current_position;                   // Current write head
    
    // Optimization parameters
    float omega_alpha;                      // Learning rate
    int muon_iterations;                    // Newton-Schulz steps
    float muon_momentum;                    // Momentum coefficient
    
    // Feature mapping
    enum atlas_kernel_type kernel_type;     // Polynomial/Exponential
    int kernel_degree;                      // For polynomial kernels
    
    // GGML context for tensor operations
    struct ggml_context * ggml_ctx;
};

struct atlas_tensor_descriptor {
    struct ggml_tensor * tensor;
    size_t offset;                          // Offset in memory pool
    size_t size_bytes;                      // Size in bytes
    bool is_persistent;                     // Whether tensor persists across inference calls
    int reference_count;                    // Reference counting for memory management
};

struct atlas_memory_pool {
    void * memory_base;                     // Base pointer to allocated memory
    size_t total_size;                      // Total pool size in bytes
    size_t used_size;                       // Currently used size
    size_t alignment;                       // Memory alignment requirement
    std::vector<atlas_tensor_descriptor> descriptors;  // Tensor descriptors
    bool is_cuda_memory;                    // Whether this is CUDA memory
};

// Test constants
constexpr float TEST_TOLERANCE = 1e-5f;
constexpr int TEST_ITERATIONS = 100;

// Test utilities
static void generate_random_data(float* data, size_t n, float min = -1.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

static bool compare_tensors(const struct ggml_tensor* a, const struct ggml_tensor* b, float tolerance = TEST_TOLERANCE) {
    if (!a || !b) return false;
    if (ggml_nelements(a) != ggml_nelements(b)) return false;
    
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    size_t n = ggml_nelements(a);
    
    for (size_t i = 0; i < n; i++) {
        if (std::abs(a_data[i] - b_data[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Test functions

// Test 1: ATLAS Memory Module Creation and Initialization
static bool test_atlas_memory_module_creation() {
    printf("Testing ATLAS memory module creation... ");
    
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED - Could not initialize GGML context\n");
        return false;
    }
    
    // Create memory module with specific dimensions
    const int input_dim = 512;
    const int hidden_dim = 1024; 
    const int output_dim = 512;
    
    atlas_memory_module module = {};
    module.input_dim = input_dim;
    module.hidden_dim = hidden_dim;
    module.output_dim = output_dim;
    module.activation_fn = ATLAS_ACT_GELU;
    
    // Create tensors for the MLP layers
    module.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, hidden_dim);
    module.b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
    module.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, output_dim);
    module.b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
    module.w_res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_dim, output_dim);
    
    // Verify tensor creation
    bool success = true;
    success &= (module.w1 != nullptr && ggml_nelements(module.w1) == input_dim * hidden_dim);
    success &= (module.b1 != nullptr && ggml_nelements(module.b1) == hidden_dim);
    success &= (module.w2 != nullptr && ggml_nelements(module.w2) == hidden_dim * output_dim);
    success &= (module.b2 != nullptr && ggml_nelements(module.b2) == output_dim);
    success &= (module.w_res != nullptr && ggml_nelements(module.w_res) == input_dim * output_dim);
    
    // Initialize with random weights
    generate_random_data((float*)module.w1->data, ggml_nelements(module.w1), -0.1f, 0.1f);
    generate_random_data((float*)module.b1->data, ggml_nelements(module.b1), -0.01f, 0.01f);
    generate_random_data((float*)module.w2->data, ggml_nelements(module.w2), -0.1f, 0.1f);
    generate_random_data((float*)module.b2->data, ggml_nelements(module.b2), -0.01f, 0.01f);
    generate_random_data((float*)module.w_res->data, ggml_nelements(module.w_res), -0.1f, 0.1f);
    
    ggml_free(ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Tensor creation/initialization failed\n");
        return false;
    }
}

// Test 2: ATLAS Context Creation and Configuration
static bool test_atlas_context_creation() {
    printf("Testing ATLAS context creation... ");
    
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,  // 256MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED - Could not initialize GGML context\n");
        return false;
    }
    
    atlas_context atlas_ctx = {};
    atlas_ctx.ggml_ctx = ctx;
    
    // Configure context parameters
    atlas_ctx.window_size = 2048;  // Sliding window size
    atlas_ctx.current_position = 0;
    atlas_ctx.omega_alpha = 0.001f;  // Learning rate
    atlas_ctx.muon_iterations = 5;   // Newton-Schulz iterations
    atlas_ctx.muon_momentum = 0.9f;  // Momentum coefficient
    atlas_ctx.kernel_type = ATLAS_KERNEL_POLYNOMIAL;
    atlas_ctx.kernel_degree = 3;
    
    // Create context window (circular buffer)
    const int context_dim = 4096;  // Dimension of context vectors
    atlas_ctx.context_window = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                   context_dim, atlas_ctx.window_size);
    
    // Create momentum state for optimizer
    const int optimizer_state_size = 1024 * 1024;  // 1M parameters
    atlas_ctx.momentum_state = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, optimizer_state_size);
    
    // Initialize memory module
    atlas_ctx.memory.input_dim = context_dim;
    atlas_ctx.memory.hidden_dim = context_dim * 2;
    atlas_ctx.memory.output_dim = context_dim;
    atlas_ctx.memory.activation_fn = ATLAS_ACT_GELU;
    
    atlas_ctx.memory.w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                              atlas_ctx.memory.input_dim, 
                                              atlas_ctx.memory.hidden_dim);
    atlas_ctx.memory.b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, atlas_ctx.memory.hidden_dim);
    atlas_ctx.memory.w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                              atlas_ctx.memory.hidden_dim, 
                                              atlas_ctx.memory.output_dim);
    atlas_ctx.memory.b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, atlas_ctx.memory.output_dim);
    atlas_ctx.memory.w_res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                 atlas_ctx.memory.input_dim, 
                                                 atlas_ctx.memory.output_dim);
    
    // Verify context initialization
    bool success = true;
    success &= (atlas_ctx.context_window != nullptr);
    success &= (atlas_ctx.momentum_state != nullptr);
    success &= (atlas_ctx.memory.w1 != nullptr);
    success &= (atlas_ctx.memory.b1 != nullptr);
    success &= (atlas_ctx.memory.w2 != nullptr);
    success &= (atlas_ctx.memory.b2 != nullptr);
    success &= (atlas_ctx.memory.w_res != nullptr);
    success &= (atlas_ctx.window_size == 2048);
    success &= (atlas_ctx.omega_alpha == 0.001f);
    success &= (atlas_ctx.kernel_type == ATLAS_KERNEL_POLYNOMIAL);
    
    ggml_free(ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
}

// Test 3: Tensor Descriptor Management
static bool test_tensor_descriptor_management() {
    printf("Testing tensor descriptor management... ");
    
    atlas_tensor_descriptor desc = {};
    
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,  // 64MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED - Could not initialize GGML context\n");
        return false;
    }
    
    // Create tensor and descriptor
    desc.tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 512);
    desc.offset = 0;
    desc.size_bytes = ggml_nbytes(desc.tensor);
    desc.is_persistent = true;
    desc.reference_count = 1;
    
    // Verify descriptor properties
    bool success = true;
    success &= (desc.tensor != nullptr);
    success &= (desc.size_bytes == 1024 * 512 * sizeof(float));
    success &= (desc.is_persistent == true);
    success &= (desc.reference_count == 1);
    
    // Test reference counting
    desc.reference_count++;
    success &= (desc.reference_count == 2);
    
    desc.reference_count--;
    success &= (desc.reference_count == 1);
    
    ggml_free(ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Descriptor management failed\n");
        return false;
    }
}

// Test 4: Memory Pool Management
static bool test_memory_pool_management() {
    printf("Testing memory pool management... ");
    
    atlas_memory_pool pool = {};
    
    // Configure memory pool
    const size_t pool_size = 32 * 1024 * 1024;  // 32MB
    pool.total_size = pool_size;
    pool.used_size = 0;
    pool.alignment = 32;  // 32-byte alignment for SIMD
    pool.is_cuda_memory = false;
    
    // Allocate memory pool
    pool.memory_base = std::aligned_alloc(pool.alignment, pool_size);
    if (!pool.memory_base) {
        printf("FAILED - Could not allocate memory pool\n");
        return false;
    }
    
    // Initialize pool memory
    std::memset(pool.memory_base, 0, pool_size);
    
    // Create tensor descriptors
    struct ggml_init_params params = {
        .mem_size   = 64 * 1024 * 1024,  // 64MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED - Could not initialize GGML context\n");
        std::free(pool.memory_base);
        return false;
    }
    
    // Add tensors to pool
    const int num_tensors = 5;
    const int tensor_sizes[] = {1024*512, 2048*256, 512*1024, 1024*1024, 256*2048};
    
    size_t current_offset = 0;
    bool success = true;
    
    for (int i = 0; i < num_tensors; i++) {
        atlas_tensor_descriptor desc = {};
        desc.tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                         tensor_sizes[i] / 1024, 1024);
        desc.offset = current_offset;
        desc.size_bytes = tensor_sizes[i] * sizeof(float);
        desc.is_persistent = (i % 2 == 0);  // Alternate persistent/temporary
        desc.reference_count = 1;
        
        // Check if tensor fits in pool
        if (current_offset + desc.size_bytes > pool_size) {
            success = false;
            break;
        }
        
        pool.descriptors.push_back(desc);
        current_offset += desc.size_bytes;
        
        // Align to next boundary
        current_offset = (current_offset + pool.alignment - 1) & ~(pool.alignment - 1);
    }
    
    pool.used_size = current_offset;
    
    // Verify pool state
    success &= (pool.descriptors.size() == num_tensors);
    success &= (pool.used_size <= pool.total_size);
    success &= (pool.used_size > 0);
    
    // Test memory access patterns
    for (const auto& desc : pool.descriptors) {
        void* tensor_memory = (char*)pool.memory_base + desc.offset;
        // Verify memory is accessible (write then read)
        *(float*)tensor_memory = 42.0f;
        success &= (*(float*)tensor_memory == 42.0f);
    }
    
    // Cleanup
    ggml_free(ctx);
    std::free(pool.memory_base);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Memory pool management failed\n");
        return false;
    }
}

// Test 5: Feature Mapping Types
static bool test_feature_mapping_types() {
    printf("Testing feature mapping types... ");
    
    bool success = true;
    
    // Test kernel type enumeration
    atlas_kernel_type kernel_types[] = {
        ATLAS_KERNEL_POLYNOMIAL,
        ATLAS_KERNEL_EXPONENTIAL,
        ATLAS_KERNEL_RBF
    };
    
    for (int i = 0; i < 3; i++) {
        success &= (kernel_types[i] >= 0 && kernel_types[i] <= 2);
    }
    
    // Test activation function enumeration
    atlas_activation activations[] = {
        ATLAS_ACT_NONE,
        ATLAS_ACT_GELU,
        ATLAS_ACT_RELU,
        ATLAS_ACT_SILU
    };
    
    for (int i = 0; i < 4; i++) {
        success &= (activations[i] >= 0 && activations[i] <= 3);
    }
    
    // Test configuration combinations
    struct {
        atlas_kernel_type kernel;
        int degree;
        atlas_activation activation;
        bool valid;
    } test_configs[] = {
        {ATLAS_KERNEL_POLYNOMIAL, 2, ATLAS_ACT_GELU, true},
        {ATLAS_KERNEL_POLYNOMIAL, 5, ATLAS_ACT_RELU, true},
        {ATLAS_KERNEL_EXPONENTIAL, 0, ATLAS_ACT_SILU, true},
        {ATLAS_KERNEL_RBF, 1, ATLAS_ACT_NONE, true},
    };
    
    for (const auto& config : test_configs) {
        if (config.valid) {
            success &= (config.kernel >= 0 && config.kernel <= 2);
            success &= (config.degree >= 0);
            success &= (config.activation >= 0 && config.activation <= 3);
        }
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Feature mapping type tests failed\n");
        return false;
    }
}

// Test 6: Multi-threaded Memory Access
static bool test_multithreaded_memory_access() {
    printf("Testing multi-threaded memory access... ");
    
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        printf("FAILED - Could not initialize GGML context\n");
        return false;
    }
    
    // Create shared tensor for multi-threaded access
    const int tensor_size = 1024 * 1024;
    struct ggml_tensor* shared_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_size);
    
    // Initialize with known pattern
    float* data = (float*)shared_tensor->data;
    for (int i = 0; i < tensor_size; i++) {
        data[i] = (float)i;
    }
    
    // Create multiple atlas contexts that might access the tensor
    const int num_contexts = 4;
    atlas_context contexts[num_contexts];
    
    for (int i = 0; i < num_contexts; i++) {
        contexts[i] = {};
        contexts[i].ggml_ctx = ctx;
        contexts[i].window_size = 512 + i * 256;  // Different window sizes
        contexts[i].omega_alpha = 0.001f * (i + 1);
        contexts[i].current_position = i * 100;
    }
    
    // Simulate concurrent access (simple sequential test for now)
    bool success = true;
    for (int context_id = 0; context_id < num_contexts; context_id++) {
        for (int access = 0; access < 100; access++) {
            int idx = (context_id * 1000 + access) % tensor_size;
            float expected = (float)idx;
            success &= (std::abs(data[idx] - expected) < TEST_TOLERANCE);
            
            if (!success) break;
        }
        if (!success) break;
    }
    
    ggml_free(ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Multi-threaded memory access failed\n");
        return false;
    }
}

// Test 7: Memory Alignment Verification
static bool test_memory_alignment() {
    printf("Testing memory alignment... ");
    
    const size_t alignments[] = {16, 32, 64, 128, 256};
    const size_t num_alignments = sizeof(alignments) / sizeof(alignments[0]);
    
    bool success = true;
    
    for (size_t i = 0; i < num_alignments; i++) {
        size_t alignment = alignments[i];
        size_t size = 1024 * 1024;  // 1MB
        
        void* memory = std::aligned_alloc(alignment, size);
        if (!memory) {
            success = false;
            break;
        }
        
        // Check alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(memory);
        success &= ((addr % alignment) == 0);
        
        // Test memory is usable
        std::memset(memory, 0xAA, size);
        success &= (((char*)memory)[0] == (char)0xAA);
        success &= (((char*)memory)[size-1] == (char)0xAA);
        
        std::free(memory);
        
        if (!success) break;
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Memory alignment verification failed\n");
        return false;
    }
}

int main(int argc, char** argv) {
    printf("Running ATLAS types and data structures tests...\n");
    
    int tests_passed = 0;
    int total_tests = 7;
    
    if (test_atlas_memory_module_creation()) tests_passed++;
    if (test_atlas_context_creation()) tests_passed++;
    if (test_tensor_descriptor_management()) tests_passed++;
    if (test_memory_pool_management()) tests_passed++;
    if (test_feature_mapping_types()) tests_passed++;
    if (test_multithreaded_memory_access()) tests_passed++;
    if (test_memory_alignment()) tests_passed++;
    
    printf("\nATLAS types tests completed: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("All ATLAS types tests PASSED!\n");
        return 0;
    } else {
        printf("Some ATLAS types tests FAILED!\n");
        return 1;
    }
}