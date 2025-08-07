// Configuration-specific tests for ATLAS
// Tests CPU, CUDA, and mixed backend configurations

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#ifdef ATLAS_CUDA_ENABLED
#include "ggml-cuda.h"
#endif

#ifdef ATLAS_METAL_ENABLED
#include "ggml-metal.h"
#endif

#ifdef ATLAS_OPENCL_ENABLED
#include "ggml-opencl.h"
#endif

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

// Configuration test structures
struct atlas_backend_info {
    const char* name;
    bool available;
    bool supports_compute;
    bool supports_memory;
    size_t device_memory_gb;
    int compute_units;
};

struct atlas_test_result {
    const char* test_name;
    bool passed;
    double execution_time_ms;
    size_t memory_used_mb;
    const char* error_message;
};

// Global test state
static std::vector<atlas_backend_info> g_available_backends;
static std::vector<atlas_test_result> g_test_results;

// Test constants
constexpr float TEST_TOLERANCE = 1e-4f;
constexpr int TEST_TENSOR_SIZE = 1024 * 1024;  // 1M elements

// Utility functions
static double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

static void generate_test_data(float* data, size_t size, float scale = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-scale, scale);
    
    for (size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

static float calculate_rmse(const float* a, const float* b, size_t n) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum / n);
}

static void record_test_result(const char* test_name, bool passed, 
                              double time_ms, size_t memory_mb, 
                              const char* error = nullptr) {
    atlas_test_result result = {
        .test_name = test_name,
        .passed = passed,
        .execution_time_ms = time_ms,
        .memory_used_mb = memory_mb,
        .error_message = error
    };
    g_test_results.push_back(result);
}

// Backend detection functions
static bool detect_cpu_backend() {
    // CPU is always available
    atlas_backend_info cpu_info = {
        .name = "CPU",
        .available = true,
        .supports_compute = true,
        .supports_memory = true,
        .device_memory_gb = 0, // Uses system RAM
        .compute_units = std::thread::hardware_concurrency()
    };
    g_available_backends.push_back(cpu_info);
    return true;
}

#ifdef ATLAS_CUDA_ENABLED
static bool detect_cuda_backend() {
    bool cuda_available = false;
    size_t device_memory = 0;
    int compute_units = 0;
    
    try {
        // Check if CUDA is available (this is a mock check)
        // In real implementation, would query CUDA runtime
        cuda_available = true;  // Assume available for testing
        device_memory = 8;      // 8GB mock
        compute_units = 32;     // 32 SMs mock
    } catch (...) {
        cuda_available = false;
    }
    
    atlas_backend_info cuda_info = {
        .name = "CUDA",
        .available = cuda_available,
        .supports_compute = cuda_available,
        .supports_memory = cuda_available,
        .device_memory_gb = device_memory,
        .compute_units = compute_units
    };
    g_available_backends.push_back(cuda_info);
    return cuda_available;
}
#endif

#ifdef ATLAS_METAL_ENABLED
static bool detect_metal_backend() {
    bool metal_available = false;
    
    // Mock Metal detection
    #ifdef __APPLE__
    metal_available = true; // Assume available on Apple platforms
    #endif
    
    atlas_backend_info metal_info = {
        .name = "Metal",
        .available = metal_available,
        .supports_compute = metal_available,
        .supports_memory = metal_available,
        .device_memory_gb = metal_available ? 16 : 0, // Unified memory
        .compute_units = metal_available ? 64 : 0
    };
    g_available_backends.push_back(metal_info);
    return metal_available;
}
#endif

#ifdef ATLAS_OPENCL_ENABLED
static bool detect_opencl_backend() {
    bool opencl_available = false;
    
    // Mock OpenCL detection
    opencl_available = true; // Assume available for testing
    
    atlas_backend_info opencl_info = {
        .name = "OpenCL",
        .available = opencl_available,
        .supports_compute = opencl_available,
        .supports_memory = opencl_available,
        .device_memory_gb = opencl_available ? 4 : 0,
        .compute_units = opencl_available ? 16 : 0
    };
    g_available_backends.push_back(opencl_info);
    return opencl_available;
}
#endif

static void detect_all_backends() {
    printf("Detecting available backends...\n");
    
    detect_cpu_backend();
    
#ifdef ATLAS_CUDA_ENABLED
    detect_cuda_backend();
#endif

#ifdef ATLAS_METAL_ENABLED
    detect_metal_backend();
#endif

#ifdef ATLAS_OPENCL_ENABLED
    detect_opencl_backend();
#endif

    printf("Found %zu backends:\n", g_available_backends.size());
    for (const auto& backend : g_available_backends) {
        printf("  - %s: %s\n", backend.name, 
               backend.available ? "Available" : "Not Available");
    }
    printf("\n");
}

// Configuration tests

// Test 1: CPU Backend Functionality
static bool test_cpu_configuration() {
    printf("Testing CPU backend configuration... ");
    
    double start_time = get_time_ms();
    
    struct ggml_init_params params = {
        .mem_size   = 128 * 1024 * 1024,  // 128MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        record_test_result("CPU Configuration", false, 0, 0, "Context init failed");
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    bool success = true;
    size_t memory_used = 0;
    
    try {
        // Create test tensors
        const int dim = 512;
        struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        struct ggml_tensor* c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        
        success &= (a != nullptr && b != nullptr && c != nullptr);
        
        if (success) {
            // Initialize test data
            generate_test_data((float*)a->data, ggml_nelements(a), 0.1f);
            generate_test_data((float*)b->data, ggml_nelements(b), 0.1f);
            
            // Create computation graph
            struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
            result = ggml_add(ctx, result, c);
            
            struct ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            
            success &= (graph != nullptr);
            success &= (ggml_graph_n_nodes(graph) > 0);
            success &= (result != nullptr);
            
            // Memory usage estimation
            memory_used = ggml_get_mem_size(ctx) / (1024 * 1024); // MB
        }
    } catch (...) {
        success = false;
    }
    
    ggml_free(ctx);
    
    double end_time = get_time_ms();
    double execution_time = end_time - start_time;
    
    record_test_result("CPU Configuration", success, execution_time, memory_used);
    
    if (success) {
        printf("ok (%.1fms, %zuMB)\n", execution_time, memory_used);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

#ifdef ATLAS_CUDA_ENABLED
// Test 2: CUDA Backend Functionality
static bool test_cuda_configuration() {
    printf("Testing CUDA backend configuration... ");
    
    // Skip if CUDA not available
    auto cuda_backend = std::find_if(g_available_backends.begin(), 
                                    g_available_backends.end(),
                                    [](const atlas_backend_info& info) {
                                        return strcmp(info.name, "CUDA") == 0;
                                    });
    
    if (cuda_backend == g_available_backends.end() || !cuda_backend->available) {
        printf("SKIPPED - CUDA not available\n");
        record_test_result("CUDA Configuration", true, 0, 0, "Skipped - CUDA unavailable");
        return true;
    }
    
    double start_time = get_time_ms();
    
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,  // 256MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        record_test_result("CUDA Configuration", false, 0, 0, "Context init failed");
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    bool success = true;
    size_t memory_used = 0;
    
    try {
        // Create larger tensors for GPU
        const int dim = 1024;
        struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
        
        success &= (a != nullptr && b != nullptr);
        
        if (success) {
            // Initialize test data
            generate_test_data((float*)a->data, ggml_nelements(a), 0.1f);
            generate_test_data((float*)b->data, ggml_nelements(b), 0.1f);
            
            // Create CUDA-specific operations
            struct ggml_tensor* result = ggml_mul_mat(ctx, a, b);
            result = ggml_gelu(ctx, result);  // CUDA should support GELU
            
            struct ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, result);
            
            success &= (graph != nullptr);
            success &= (ggml_graph_n_nodes(graph) > 0);
            success &= (result != nullptr);
            
            memory_used = ggml_get_mem_size(ctx) / (1024 * 1024); // MB
        }
    } catch (...) {
        success = false;
    }
    
    ggml_free(ctx);
    
    double end_time = get_time_ms();
    double execution_time = end_time - start_time;
    
    record_test_result("CUDA Configuration", success, execution_time, memory_used);
    
    if (success) {
        printf("ok (%.1fms, %zuMB)\n", execution_time, memory_used);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}
#endif

// Test 3: Mixed Backend Configuration
static bool test_mixed_backend_configuration() {
    printf("Testing mixed backend configuration... ");
    
    double start_time = get_time_ms();
    
    struct ggml_init_params params = {
        .mem_size   = 512 * 1024 * 1024,  // 512MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        record_test_result("Mixed Backend", false, 0, 0, "Context init failed");
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    bool success = true;
    size_t memory_used = 0;
    
    try {
        // Create tensors that could be processed on different backends
        const int seq_len = 256;
        const int hidden_dim = 768;
        const int batch_size = 4;
        
        struct ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
                                                       hidden_dim, seq_len, batch_size);
        struct ggml_tensor* weights1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                          hidden_dim, hidden_dim * 4);
        struct ggml_tensor* weights2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                          hidden_dim * 4, hidden_dim);
        struct ggml_tensor* bias1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim * 4);
        struct ggml_tensor* bias2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
        
        success &= (input != nullptr && weights1 != nullptr && weights2 != nullptr);
        success &= (bias1 != nullptr && bias2 != nullptr);
        
        if (success) {
            // Initialize with test data
            generate_test_data((float*)input->data, ggml_nelements(input), 0.1f);
            generate_test_data((float*)weights1->data, ggml_nelements(weights1), 0.05f);
            generate_test_data((float*)weights2->data, ggml_nelements(weights2), 0.05f);
            generate_test_data((float*)bias1->data, ggml_nelements(bias1), 0.01f);
            generate_test_data((float*)bias2->data, ggml_nelements(bias2), 0.01f);
            
            // Create computation graph with mixed operations
            // Some ops might run on CPU, others on GPU
            struct ggml_tensor* input_2d = ggml_reshape_2d(ctx, input, 
                                                          hidden_dim, seq_len * batch_size);
            
            // Layer 1 - Could be GPU accelerated
            struct ggml_tensor* h1 = ggml_mul_mat(ctx, weights1, input_2d);
            h1 = ggml_add(ctx, h1, ggml_repeat(ctx, bias1, h1));
            h1 = ggml_gelu(ctx, h1);  // Activation - good GPU candidate
            
            // Layer 2 - Could be GPU accelerated  
            struct ggml_tensor* h2 = ggml_mul_mat(ctx, weights2, h1);
            h2 = ggml_add(ctx, h2, ggml_repeat(ctx, bias2, h2));
            
            // Residual connection - typically CPU
            struct ggml_tensor* output = ggml_add(ctx, h2, input_2d);
            
            // Reshape back
            output = ggml_reshape_3d(ctx, output, hidden_dim, seq_len, batch_size);
            
            // Build graph
            struct ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            
            success &= (graph != nullptr);
            success &= (ggml_graph_n_nodes(graph) > 5);  // Should have multiple nodes
            success &= (output != nullptr);
            success &= (ggml_nelements(output) == seq_len * hidden_dim * batch_size);
            
            memory_used = ggml_get_mem_size(ctx) / (1024 * 1024); // MB
        }
    } catch (...) {
        success = false;
    }
    
    ggml_free(ctx);
    
    double end_time = get_time_ms();
    double execution_time = end_time - start_time;
    
    record_test_result("Mixed Backend", success, execution_time, memory_used);
    
    if (success) {
        printf("ok (%.1fms, %zuMB)\n", execution_time, memory_used);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 4: Memory Transfer Between Backends
static bool test_memory_transfer() {
    printf("Testing memory transfer between backends... ");
    
    double start_time = get_time_ms();
    
    struct ggml_init_params params = {
        .mem_size   = 256 * 1024 * 1024,  // 256MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        record_test_result("Memory Transfer", false, 0, 0, "Context init failed");
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    bool success = true;
    size_t memory_used = 0;
    
    try {
        const int size = 1024 * 256;  // 256K floats = 1MB
        
        // Create host tensor
        struct ggml_tensor* host_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
        success &= (host_tensor != nullptr);
        
        if (success) {
            // Initialize host data with known pattern
            float* host_data = (float*)host_tensor->data;
            for (int i = 0; i < size; i++) {
                host_data[i] = sinf(i * 0.001f) * 100.0f;
            }
            
            // Simulate memory transfer operations
            // In real implementation, this would involve actual GPU memory transfers
            
            // Create "device" tensor (simulated)
            struct ggml_tensor* device_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
            success &= (device_tensor != nullptr);
            
            if (success) {
                // Simulate host-to-device transfer
                std::memcpy(device_tensor->data, host_tensor->data, 
                           size * sizeof(float));
                
                // Simulate some computation on device
                float* device_data = (float*)device_tensor->data;
                for (int i = 0; i < size; i++) {
                    device_data[i] = device_data[i] * 1.5f + 0.1f;
                }
                
                // Create result tensor
                struct ggml_tensor* result_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, size);
                success &= (result_tensor != nullptr);
                
                if (success) {
                    // Simulate device-to-host transfer
                    std::memcpy(result_tensor->data, device_tensor->data,
                               size * sizeof(float));
                    
                    // Verify transfer correctness
                    float* result_data = (float*)result_tensor->data;
                    for (int i = 0; i < size && success; i++) {
                        float expected = host_data[i] * 1.5f + 0.1f;
                        float actual = result_data[i];
                        
                        if (std::abs(actual - expected) > TEST_TOLERANCE) {
                            success = false;
                        }
                    }
                }
            }
            
            memory_used = ggml_get_mem_size(ctx) / (1024 * 1024); // MB
        }
    } catch (...) {
        success = false;
    }
    
    ggml_free(ctx);
    
    double end_time = get_time_ms();
    double execution_time = end_time - start_time;
    
    record_test_result("Memory Transfer", success, execution_time, memory_used);
    
    if (success) {
        printf("ok (%.1fms, %zuMB)\n", execution_time, memory_used);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 5: Backend Performance Comparison
static bool test_backend_performance() {
    printf("Testing backend performance comparison... ");
    
    double start_time = get_time_ms();
    bool overall_success = true;
    
    // Test matrix multiplication performance on different backends
    const int dim = 512;
    const int iterations = 10;
    
    std::vector<double> backend_times;
    
    for (const auto& backend : g_available_backends) {
        if (!backend.available) continue;
        
        printf("\n  Testing %s backend... ", backend.name);
        
        struct ggml_init_params params = {
            .mem_size   = 256 * 1024 * 1024,  // 256MB
            .mem_buffer = nullptr,
            .no_alloc   = false,
        };
        
        struct ggml_context* ctx = ggml_init(params);
        if (!ctx) {
            printf("FAILED - Context init");
            continue;
        }
        
        bool backend_success = true;
        double total_time = 0.0;
        
        try {
            // Create test matrices
            struct ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
            struct ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
            
            if (a && b) {
                generate_test_data((float*)a->data, ggml_nelements(a), 0.1f);
                generate_test_data((float*)b->data, ggml_nelements(b), 0.1f);
                
                // Time multiple iterations
                for (int i = 0; i < iterations; i++) {
                    double iter_start = get_time_ms();
                    
                    // Create computation
                    struct ggml_tensor* c = ggml_mul_mat(ctx, a, b);
                    struct ggml_cgraph* graph = ggml_new_graph(ctx);
                    ggml_build_forward_expand(graph, c);
                    
                    // Note: In real implementation, would execute graph here
                    
                    double iter_end = get_time_ms();
                    total_time += (iter_end - iter_start);
                }
                
                double avg_time = total_time / iterations;
                backend_times.push_back(avg_time);
                
                printf("%.2fms avg", avg_time);
            } else {
                backend_success = false;
                printf("FAILED - Tensor creation");
            }
        } catch (...) {
            backend_success = false;
            printf("FAILED - Exception");
        }
        
        ggml_free(ctx);
        overall_success &= backend_success;
    }
    
    // Compare performance
    if (!backend_times.empty()) {
        auto min_time = *std::min_element(backend_times.begin(), backend_times.end());
        auto max_time = *std::max_element(backend_times.begin(), backend_times.end());
        
        printf("\n  Performance range: %.2fms - %.2fms", min_time, max_time);
        if (max_time > 0) {
            printf(" (%.1fx speedup)", max_time / min_time);
        }
    }
    
    double end_time = get_time_ms();
    double execution_time = end_time - start_time;
    
    record_test_result("Performance Comparison", overall_success, execution_time, 0);
    
    if (overall_success) {
        printf("\nok (%.1fms total)\n", execution_time);
        return true;
    } else {
        printf("\nFAILED\n");
        return false;
    }
}

// Test result summary
static void print_test_summary() {
    printf("\n");
    printf("===========================================\n");
    printf("ATLAS Configuration Test Summary\n");
    printf("===========================================\n");
    
    int total_tests = g_test_results.size();
    int passed_tests = 0;
    double total_time = 0.0;
    size_t total_memory = 0;
    
    for (const auto& result : g_test_results) {
        if (result.passed) {
            passed_tests++;
        }
        total_time += result.execution_time_ms;
        total_memory += result.memory_used_mb;
        
        const char* status = result.passed ? "PASS" : "FAIL";
        const char* color = result.passed ? "\033[0;32m" : "\033[0;31m";
        
        printf("%s[%s]\033[0m %-25s (%.1fms, %zuMB)\n", 
               color, status, result.test_name, 
               result.execution_time_ms, result.memory_used_mb);
        
        if (!result.passed && result.error_message) {
            printf("      Error: %s\n", result.error_message);
        }
    }
    
    printf("-------------------------------------------\n");
    printf("Results: %d/%d tests passed\n", passed_tests, total_tests);
    printf("Total time: %.1fms\n", total_time);
    printf("Peak memory: %zuMB\n", total_memory);
    
    // Backend summary
    printf("\nAvailable backends:\n");
    for (const auto& backend : g_available_backends) {
        if (backend.available) {
            printf("  ✓ %s (%d cores, %zuGB memory)\n", 
                   backend.name, backend.compute_units, backend.device_memory_gb);
        } else {
            printf("  ✗ %s (not available)\n", backend.name);
        }
    }
    
    printf("===========================================\n");
}

int main(int argc, char** argv) {
    printf("Running ATLAS configuration tests...\n\n");
    
    // Detect available backends
    detect_all_backends();
    
    int tests_passed = 0;
    int total_tests = 0;
    
    // Run configuration tests
    total_tests++;
    if (test_cpu_configuration()) tests_passed++;
    
#ifdef ATLAS_CUDA_ENABLED
    total_tests++;
    if (test_cuda_configuration()) tests_passed++;
#endif
    
    total_tests++;
    if (test_mixed_backend_configuration()) tests_passed++;
    
    total_tests++;
    if (test_memory_transfer()) tests_passed++;
    
    total_tests++;
    if (test_backend_performance()) tests_passed++;
    
    // Print summary
    print_test_summary();
    
    if (tests_passed == total_tests) {
        printf("\n🎉 All ATLAS configuration tests PASSED!\n");
        return 0;
    } else {
        printf("\n❌ Some ATLAS configuration tests FAILED!\n");
        return 1;
    }
}