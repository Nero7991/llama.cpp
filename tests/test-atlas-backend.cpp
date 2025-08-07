// Unit tests for ATLAS backend implementation
// Tests backend registry, operations, and integration with GGML

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
#include <chrono>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ATLAS backend definitions
enum atlas_backend_type {
    ATLAS_BACKEND_CPU = 0,
    ATLAS_BACKEND_CUDA,
    ATLAS_BACKEND_OPENCL,
    ATLAS_BACKEND_METAL,
    ATLAS_BACKEND_COUNT
};

struct atlas_backend_config {
    atlas_backend_type type;
    int device_id;
    size_t memory_limit;     // Memory limit in bytes
    bool enable_optimization; // Enable backend-specific optimizations
    int compute_units;       // Number of compute units/SMs
    const char* device_name;
};

struct atlas_operation {
    const char* name;
    ggml_op ggml_op_type;
    bool (*is_supported)(const struct ggml_tensor* tensor);
    void (*execute)(struct ggml_tensor* dst, const struct ggml_tensor* src);
    size_t (*get_memory_requirement)(const struct ggml_tensor* tensor);
};

struct atlas_backend_interface {
    atlas_backend_type type;
    const char* name;
    
    // Backend lifecycle
    bool (*init)(atlas_backend_config* config);
    void (*shutdown)();
    
    // Memory management
    void* (*alloc_memory)(size_t size, size_t alignment);
    void (*free_memory)(void* ptr);
    void (*sync_device)();
    
    // Operation support
    bool (*supports_op)(ggml_op op);
    const atlas_operation* (*get_operation)(ggml_op op);
    
    // Performance metrics
    double (*get_flops_per_second)();
    size_t (*get_memory_bandwidth)();
};

// Mock backend implementations for testing
static atlas_backend_interface g_cpu_backend;
static atlas_backend_interface g_cuda_backend;
static std::vector<atlas_backend_interface*> g_registered_backends;

// Test constants
constexpr float TEST_TOLERANCE = 1e-5f;
constexpr size_t TEST_MEMORY_SIZE = 64 * 1024 * 1024; // 64MB

// Mock CPU backend implementation
static bool cpu_backend_init(atlas_backend_config* config) {
    return config && config->type == ATLAS_BACKEND_CPU;
}

static void cpu_backend_shutdown() {
    // No-op for CPU
}

static void* cpu_backend_alloc(size_t size, size_t alignment) {
    return std::aligned_alloc(alignment, size);
}

static void cpu_backend_free(void* ptr) {
    std::free(ptr);
}

static void cpu_backend_sync() {
    // No-op for CPU
}

static bool cpu_supports_op(ggml_op op) {
    // CPU backend supports most basic operations
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_GELU:
        case GGML_OP_RELU:
        case GGML_OP_SILU:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
            return true;
        default:
            return false;
    }
}

static const atlas_operation* cpu_get_operation(ggml_op op) {
    static atlas_operation cpu_ops[] = {
        {"cpu_add", GGML_OP_ADD, nullptr, nullptr, nullptr},
        {"cpu_mul", GGML_OP_MUL, nullptr, nullptr, nullptr},
        {"cpu_mul_mat", GGML_OP_MUL_MAT, nullptr, nullptr, nullptr},
    };
    
    for (auto& cpu_op : cpu_ops) {
        if (cpu_op.ggml_op_type == op) {
            return &cpu_op;
        }
    }
    return nullptr;
}

static double cpu_get_flops() {
    return 100.0e9; // 100 GFLOPS (mock)
}

static size_t cpu_get_bandwidth() {
    return 50.0e9; // 50 GB/s (mock)
}

// Mock CUDA backend implementation
static bool cuda_backend_init(atlas_backend_config* config) {
    return config && config->type == ATLAS_BACKEND_CUDA;
}

static void cuda_backend_shutdown() {
    // Mock CUDA cleanup
}

static void* cuda_backend_alloc(size_t size, size_t alignment) {
    // Mock CUDA memory allocation
    return std::aligned_alloc(alignment, size);
}

static void cuda_backend_free(void* ptr) {
    std::free(ptr);
}

static void cuda_backend_sync() {
    // Mock CUDA synchronization
}

static bool cuda_supports_op(ggml_op op) {
    // CUDA backend supports more operations
    switch (op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_MUL_MAT:
        case GGML_OP_GELU:
        case GGML_OP_RELU:
        case GGML_OP_SILU:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_RMS_NORM:
        case GGML_OP_FLASH_ATTN_EXT:
            return true;
        default:
            return false;
    }
}

static const atlas_operation* cuda_get_operation(ggml_op op) {
    static atlas_operation cuda_ops[] = {
        {"cuda_add", GGML_OP_ADD, nullptr, nullptr, nullptr},
        {"cuda_mul", GGML_OP_MUL, nullptr, nullptr, nullptr},
        {"cuda_mul_mat", GGML_OP_MUL_MAT, nullptr, nullptr, nullptr},
        {"cuda_flash_attn", GGML_OP_FLASH_ATTN_EXT, nullptr, nullptr, nullptr},
    };
    
    for (auto& cuda_op : cuda_ops) {
        if (cuda_op.ggml_op_type == op) {
            return &cuda_op;
        }
    }
    return nullptr;
}

static double cuda_get_flops() {
    return 10000.0e9; // 10 TFLOPS (mock)
}

static size_t cuda_get_bandwidth() {
    return 900.0e9; // 900 GB/s (mock)
}

// Initialize mock backends
static void init_mock_backends() {
    // CPU backend
    g_cpu_backend = {
        .type = ATLAS_BACKEND_CPU,
        .name = "ATLAS CPU Backend",
        .init = cpu_backend_init,
        .shutdown = cpu_backend_shutdown,
        .alloc_memory = cpu_backend_alloc,
        .free_memory = cpu_backend_free,
        .sync_device = cpu_backend_sync,
        .supports_op = cpu_supports_op,
        .get_operation = cpu_get_operation,
        .get_flops_per_second = cpu_get_flops,
        .get_memory_bandwidth = cpu_get_bandwidth,
    };
    
    // CUDA backend
    g_cuda_backend = {
        .type = ATLAS_BACKEND_CUDA,
        .name = "ATLAS CUDA Backend",
        .init = cuda_backend_init,
        .shutdown = cuda_backend_shutdown,
        .alloc_memory = cuda_backend_alloc,
        .free_memory = cuda_backend_free,
        .sync_device = cuda_backend_sync,
        .supports_op = cuda_supports_op,
        .get_operation = cuda_get_operation,
        .get_flops_per_second = cuda_get_flops,
        .get_memory_bandwidth = cuda_get_bandwidth,
    };
}

// Backend registry functions
static bool register_backend(atlas_backend_interface* backend) {
    if (!backend) return false;
    
    // Check if backend is already registered
    for (auto* existing : g_registered_backends) {
        if (existing->type == backend->type) {
            return false; // Already registered
        }
    }
    
    g_registered_backends.push_back(backend);
    return true;
}

static atlas_backend_interface* get_backend(atlas_backend_type type) {
    for (auto* backend : g_registered_backends) {
        if (backend->type == type) {
            return backend;
        }
    }
    return nullptr;
}

static void clear_registry() {
    g_registered_backends.clear();
}

// Test utility functions
static void generate_random_data(float* data, size_t n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

// Test functions

// Test 1: Backend Registration
static bool test_backend_registration() {
    printf("Testing backend registration... ");
    
    clear_registry();
    init_mock_backends();
    
    // Register backends
    bool cpu_registered = register_backend(&g_cpu_backend);
    bool cuda_registered = register_backend(&g_cuda_backend);
    
    // Try to register same backend again (should fail)
    bool cpu_duplicate = register_backend(&g_cpu_backend);
    
    bool success = true;
    success &= cpu_registered;
    success &= cuda_registered;
    success &= !cpu_duplicate; // Should fail
    success &= (g_registered_backends.size() == 2);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 2: Backend Discovery and Selection
static bool test_backend_discovery() {
    printf("Testing backend discovery... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    // Test backend retrieval
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    atlas_backend_interface* missing_backend = get_backend(ATLAS_BACKEND_OPENCL);
    
    bool success = true;
    success &= (cpu_backend != nullptr);
    success &= (cuda_backend != nullptr);
    success &= (missing_backend == nullptr);
    success &= (cpu_backend->type == ATLAS_BACKEND_CPU);
    success &= (cuda_backend->type == ATLAS_BACKEND_CUDA);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 3: Backend Initialization and Configuration
static bool test_backend_initialization() {
    printf("Testing backend initialization... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    // Test CPU backend initialization
    atlas_backend_config cpu_config = {
        .type = ATLAS_BACKEND_CPU,
        .device_id = 0,
        .memory_limit = TEST_MEMORY_SIZE,
        .enable_optimization = true,
        .compute_units = 8,
        .device_name = "Test CPU"
    };
    
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    bool cpu_init_success = cpu_backend && cpu_backend->init(&cpu_config);
    
    // Test CUDA backend initialization
    atlas_backend_config cuda_config = {
        .type = ATLAS_BACKEND_CUDA,
        .device_id = 0,
        .memory_limit = TEST_MEMORY_SIZE * 4,
        .enable_optimization = true,
        .compute_units = 32,
        .device_name = "Test GPU"
    };
    
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    bool cuda_init_success = cuda_backend && cuda_backend->init(&cuda_config);
    
    // Test invalid configuration
    atlas_backend_config invalid_config = {
        .type = ATLAS_BACKEND_CPU,
        .device_id = -1,
        .memory_limit = 0,
        .enable_optimization = false,
        .compute_units = 0,
        .device_name = nullptr
    };
    
    bool invalid_init = cpu_backend->init(&invalid_config);
    
    bool success = true;
    success &= cpu_init_success;
    success &= cuda_init_success;
    success &= invalid_init; // Our mock should still accept it
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 4: Operation Support Discovery
static bool test_operation_support() {
    printf("Testing operation support discovery... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    
    // Test operation support queries
    ggml_op test_ops[] = {
        GGML_OP_ADD,
        GGML_OP_MUL_MAT,
        GGML_OP_FLASH_ATTN_EXT,
        GGML_OP_CONV_1D,  // Unlikely to be supported
    };
    
    bool success = true;
    
    for (ggml_op op : test_ops) {
        bool cpu_supports = cpu_backend->supports_op(op);
        bool cuda_supports = cuda_backend->supports_op(op);
        
        // CUDA should support more ops than CPU
        if (op == GGML_OP_FLASH_ATTN_EXT) {
            success &= !cpu_supports;  // CPU shouldn't support flash attention
            success &= cuda_supports;  // CUDA should support it
        }
        
        if (op == GGML_OP_ADD || op == GGML_OP_MUL_MAT) {
            success &= cpu_supports;   // Both should support basic ops
            success &= cuda_supports;
        }
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 5: Memory Management
static bool test_memory_management() {
    printf("Testing memory management... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    
    bool success = true;
    
    // Test CPU memory allocation
    const size_t test_sizes[] = {1024, 4096, 16384, 65536};
    const size_t alignments[] = {16, 32, 64};
    
    for (size_t size : test_sizes) {
        for (size_t alignment : alignments) {
            // CPU allocation
            void* cpu_memory = cpu_backend->alloc_memory(size, alignment);
            success &= (cpu_memory != nullptr);
            
            // Check alignment
            if (cpu_memory) {
                uintptr_t addr = reinterpret_cast<uintptr_t>(cpu_memory);
                success &= ((addr % alignment) == 0);
                
                // Test memory is usable
                std::memset(cpu_memory, 0x42, size);
                success &= (((char*)cpu_memory)[0] == 0x42);
                success &= (((char*)cpu_memory)[size-1] == 0x42);
                
                cpu_backend->free_memory(cpu_memory);
            }
            
            // CUDA allocation
            void* cuda_memory = cuda_backend->alloc_memory(size, alignment);
            success &= (cuda_memory != nullptr);
            
            if (cuda_memory) {
                uintptr_t addr = reinterpret_cast<uintptr_t>(cuda_memory);
                success &= ((addr % alignment) == 0);
                
                // Test memory is usable
                std::memset(cuda_memory, 0x84, size);
                success &= (((char*)cuda_memory)[0] == (char)0x84);
                success &= (((char*)cuda_memory)[size-1] == (char)0x84);
                
                cuda_backend->free_memory(cuda_memory);
            }
            
            if (!success) break;
        }
        if (!success) break;
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 6: Performance Metrics
static bool test_performance_metrics() {
    printf("Testing performance metrics... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    
    // Get performance metrics
    double cpu_flops = cpu_backend->get_flops_per_second();
    size_t cpu_bandwidth = cpu_backend->get_memory_bandwidth();
    double cuda_flops = cuda_backend->get_flops_per_second();
    size_t cuda_bandwidth = cuda_backend->get_memory_bandwidth();
    
    bool success = true;
    success &= (cpu_flops > 0);
    success &= (cpu_bandwidth > 0);
    success &= (cuda_flops > 0);
    success &= (cuda_bandwidth > 0);
    
    // CUDA should generally have higher performance
    success &= (cuda_flops > cpu_flops);
    success &= (cuda_bandwidth > cpu_bandwidth);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 7: Backend Selection Strategy
static bool test_backend_selection() {
    printf("Testing backend selection strategy... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    // Simple backend selection logic based on operation and performance
    auto select_best_backend = [](ggml_op op) -> atlas_backend_interface* {
        atlas_backend_interface* best = nullptr;
        double best_flops = 0.0;
        
        for (auto* backend : g_registered_backends) {
            if (backend->supports_op(op)) {
                double flops = backend->get_flops_per_second();
                if (flops > best_flops) {
                    best = backend;
                    best_flops = flops;
                }
            }
        }
        return best;
    };
    
    // Test selection for different operations
    ggml_op test_ops[] = {
        GGML_OP_ADD,
        GGML_OP_MUL_MAT,
        GGML_OP_FLASH_ATTN_EXT,
    };
    
    bool success = true;
    
    for (ggml_op op : test_ops) {
        atlas_backend_interface* selected = select_best_backend(op);
        
        if (op == GGML_OP_FLASH_ATTN_EXT) {
            // Only CUDA supports flash attention
            success &= (selected && selected->type == ATLAS_BACKEND_CUDA);
        } else {
            // For ops supported by both, CUDA should be selected (higher FLOPS)
            success &= (selected && selected->type == ATLAS_BACKEND_CUDA);
        }
    }
    
    // Test with unsupported operation
    atlas_backend_interface* unsupported = select_best_backend(GGML_OP_CONV_1D);
    success &= (unsupported == nullptr);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 8: Backend Synchronization
static bool test_backend_synchronization() {
    printf("Testing backend synchronization... ");
    
    clear_registry();
    init_mock_backends();
    register_backend(&g_cpu_backend);
    register_backend(&g_cuda_backend);
    
    atlas_backend_interface* cpu_backend = get_backend(ATLAS_BACKEND_CPU);
    atlas_backend_interface* cuda_backend = get_backend(ATLAS_BACKEND_CUDA);
    
    // Test synchronization (should not crash or hang)
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cpu_backend->sync_device();
    cuda_backend->sync_device();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Synchronization should complete quickly for mock backends
    bool success = (duration.count() < 1000); // Less than 1 second
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED - Synchronization took too long\n");
        return false;
    }
}

int main(int argc, char** argv) {
    printf("Running ATLAS backend implementation tests...\n");
    
    int tests_passed = 0;
    int total_tests = 8;
    
    if (test_backend_registration()) tests_passed++;
    if (test_backend_discovery()) tests_passed++;
    if (test_backend_initialization()) tests_passed++;
    if (test_operation_support()) tests_passed++;
    if (test_memory_management()) tests_passed++;
    if (test_performance_metrics()) tests_passed++;
    if (test_backend_selection()) tests_passed++;
    if (test_backend_synchronization()) tests_passed++;
    
    printf("\nATLAS backend tests completed: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("All ATLAS backend tests PASSED!\n");
        return 0;
    } else {
        printf("Some ATLAS backend tests FAILED!\n");
        return 1;
    }
}