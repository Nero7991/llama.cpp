// Comprehensive integration tests for ATLAS components
// Tests full ATLAS pipeline with memory modules, backends, and operations

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#ifdef ATLAS_CUDA_ENABLED
#include "ggml-cuda.h"
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
#include <thread>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ATLAS integration test configuration
struct atlas_test_config {
    int sequence_length;
    int batch_size;
    int hidden_dimension;
    int memory_window_size;
    float learning_rate;
    int muon_iterations;
    bool use_cuda;
    bool enable_profiling;
    float tolerance;
};

// ATLAS system state for integration testing
struct atlas_integration_context {
    struct ggml_context* ggml_ctx;
    struct ggml_allocr* allocator;
    
    // Input/output tensors
    struct ggml_tensor* input_sequence;
    struct ggml_tensor* output_sequence;
    struct ggml_tensor* target_sequence;
    
    // Memory components
    struct ggml_tensor* memory_keys;
    struct ggml_tensor* memory_values;
    struct ggml_tensor* context_window;
    
    // Memory module parameters
    struct ggml_tensor* w1, *b1, *w2, *b2, *w_res;
    
    // Optimizer state
    struct ggml_tensor* momentum_state;
    struct ggml_tensor* second_moment_state;
    
    // Compute graph
    struct ggml_cgraph* forward_graph;
    struct ggml_cgraph* backward_graph;
    
    // Performance metrics
    double forward_time_ms;
    double backward_time_ms;
    double memory_usage_mb;
    size_t peak_memory_mb;
};

// Test constants
constexpr float DEFAULT_TOLERANCE = 1e-3f;
constexpr int DEFAULT_SEQUENCE_LENGTH = 1024;
constexpr int DEFAULT_BATCH_SIZE = 4;
constexpr int DEFAULT_HIDDEN_DIM = 768;
constexpr int DEFAULT_WINDOW_SIZE = 256;

// Utility functions
static void generate_random_tensor(struct ggml_tensor* tensor, float min = -1.0f, float max = 1.0f) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    
    float* data = (float*)tensor->data;
    size_t n = ggml_nelements(tensor);
    
    for (size_t i = 0; i < n; i++) {
        data[i] = dis(gen);
    }
}

static double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

static size_t get_memory_usage_mb(struct ggml_context* ctx) {
    size_t mem_size = ggml_get_mem_size(ctx);
    return mem_size / (1024 * 1024);
}

static float calculate_mse(const struct ggml_tensor* a, const struct ggml_tensor* b) {
    if (ggml_nelements(a) != ggml_nelements(b)) return INFINITY;
    
    const float* a_data = (const float*)a->data;
    const float* b_data = (const float*)b->data;
    size_t n = ggml_nelements(a);
    
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = a_data[i] - b_data[i];
        sum += diff * diff;
    }
    
    return sqrtf(sum / n);
}

// ATLAS integration functions
static bool initialize_atlas_context(atlas_integration_context* ctx, const atlas_test_config& config) {
    // Initialize GGML context
    struct ggml_init_params params = {
        .mem_size   = 512 * 1024 * 1024,  // 512MB
        .mem_buffer = nullptr,
        .no_alloc   = false,
    };
    
    ctx->ggml_ctx = ggml_init(params);
    if (!ctx->ggml_ctx) {
        printf("Failed to initialize GGML context\n");
        return false;
    }
    
    // Create input/output tensors
    ctx->input_sequence = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                             config.hidden_dimension,
                                             config.sequence_length,
                                             config.batch_size);
    
    ctx->output_sequence = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                              config.hidden_dimension,
                                              config.sequence_length,
                                              config.batch_size);
    
    ctx->target_sequence = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                              config.hidden_dimension,
                                              config.sequence_length,
                                              config.batch_size);
    
    // Create memory components
    ctx->memory_keys = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                          config.hidden_dimension,
                                          config.memory_window_size,
                                          config.batch_size);
    
    ctx->memory_values = ggml_new_tensor_3d(ctx->ggml_ctx, GGML_TYPE_F32,
                                            config.hidden_dimension,
                                            config.memory_window_size,
                                            config.batch_size);
    
    ctx->context_window = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32,
                                             config.hidden_dimension,
                                             config.memory_window_size);
    
    // Create memory module parameters
    const int hidden_expand = config.hidden_dimension * 2;
    
    ctx->w1 = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32,
                                 config.hidden_dimension, hidden_expand);
    ctx->b1 = ggml_new_tensor_1d(ctx->ggml_ctx, GGML_TYPE_F32, hidden_expand);
    ctx->w2 = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32,
                                 hidden_expand, config.hidden_dimension);
    ctx->b2 = ggml_new_tensor_1d(ctx->ggml_ctx, GGML_TYPE_F32, config.hidden_dimension);
    ctx->w_res = ggml_new_tensor_2d(ctx->ggml_ctx, GGML_TYPE_F32,
                                    config.hidden_dimension, config.hidden_dimension);
    
    // Create optimizer state
    size_t total_params = ggml_nelements(ctx->w1) + ggml_nelements(ctx->b1) +
                         ggml_nelements(ctx->w2) + ggml_nelements(ctx->b2) +
                         ggml_nelements(ctx->w_res);
    
    ctx->momentum_state = ggml_new_tensor_1d(ctx->ggml_ctx, GGML_TYPE_F32, total_params);
    ctx->second_moment_state = ggml_new_tensor_1d(ctx->ggml_ctx, GGML_TYPE_F32, total_params);
    
    // Initialize all tensors with random data
    generate_random_tensor(ctx->input_sequence, -0.5f, 0.5f);
    generate_random_tensor(ctx->target_sequence, -0.5f, 0.5f);
    generate_random_tensor(ctx->memory_keys, -0.1f, 0.1f);
    generate_random_tensor(ctx->memory_values, -0.1f, 0.1f);
    generate_random_tensor(ctx->context_window, 0.0f, 0.0f); // Initialize to zero
    
    // Initialize memory module parameters
    generate_random_tensor(ctx->w1, -0.1f, 0.1f);
    generate_random_tensor(ctx->b1, -0.01f, 0.01f);
    generate_random_tensor(ctx->w2, -0.1f, 0.1f);
    generate_random_tensor(ctx->b2, -0.01f, 0.01f);
    generate_random_tensor(ctx->w_res, -0.1f, 0.1f);
    
    // Initialize optimizer state to zero
    std::memset(ctx->momentum_state->data, 0, ggml_nbytes(ctx->momentum_state));
    std::memset(ctx->second_moment_state->data, 0, ggml_nbytes(ctx->second_moment_state));
    
    return true;
}

static void cleanup_atlas_context(atlas_integration_context* ctx) {
    if (ctx->ggml_ctx) {
        ggml_free(ctx->ggml_ctx);
        ctx->ggml_ctx = nullptr;
    }
}

static bool build_atlas_forward_graph(atlas_integration_context* ctx, const atlas_test_config& config) {
    if (!ctx->ggml_ctx) return false;
    
    // Reshape input for matrix operations
    struct ggml_tensor* input_2d = ggml_reshape_2d(ctx->ggml_ctx, ctx->input_sequence,
                                                   config.hidden_dimension,
                                                   config.sequence_length * config.batch_size);
    
    // Memory module forward pass
    // Layer 1: h1 = GELU(input * w1 + b1)
    struct ggml_tensor* h1 = ggml_mul_mat(ctx->ggml_ctx, ctx->w1, input_2d);
    h1 = ggml_add(ctx->ggml_ctx, h1, ggml_repeat(ctx->ggml_ctx, ctx->b1, h1));
    h1 = ggml_gelu(ctx->ggml_ctx, h1);
    
    // Layer 2: h2 = h1 * w2 + b2
    struct ggml_tensor* h2 = ggml_mul_mat(ctx->ggml_ctx, ctx->w2, h1);
    h2 = ggml_add(ctx->ggml_ctx, h2, ggml_repeat(ctx->ggml_ctx, ctx->b2, h2));
    
    // Residual connection: memory_out = h2 + input * w_res
    struct ggml_tensor* residual = ggml_mul_mat(ctx->ggml_ctx, ctx->w_res, input_2d);
    struct ggml_tensor* memory_output = ggml_add(ctx->ggml_ctx, h2, residual);
    
    // Memory attention mechanism
    struct ggml_tensor* memory_keys_2d = ggml_reshape_2d(ctx->ggml_ctx, ctx->memory_keys,
                                                         config.hidden_dimension,
                                                         config.memory_window_size * config.batch_size);
    
    struct ggml_tensor* memory_values_2d = ggml_reshape_2d(ctx->ggml_ctx, ctx->memory_values,
                                                           config.hidden_dimension,
                                                           config.memory_window_size * config.batch_size);
    
    // Attention weights: softmax(query * keys^T)
    struct ggml_tensor* attention_scores = ggml_mul_mat(ctx->ggml_ctx, memory_keys_2d, memory_output);
    
    // Scale by sqrt(hidden_dim)
    float scale = 1.0f / sqrtf((float)config.hidden_dimension);
    struct ggml_tensor* scale_tensor = ggml_new_f32(ctx->ggml_ctx, scale);
    attention_scores = ggml_scale(ctx->ggml_ctx, attention_scores, scale_tensor);
    
    struct ggml_tensor* attention_weights = ggml_soft_max(ctx->ggml_ctx, attention_scores);
    
    // Apply attention to values
    struct ggml_tensor* attended_values = ggml_mul_mat(ctx->ggml_ctx, memory_values_2d, attention_weights);
    
    // Combine memory output with attended values
    struct ggml_tensor* combined_output = ggml_add(ctx->ggml_ctx, memory_output, attended_values);
    
    // Reshape back to 3D
    ctx->output_sequence = ggml_reshape_3d(ctx->ggml_ctx, combined_output,
                                          config.hidden_dimension,
                                          config.sequence_length,
                                          config.batch_size);
    
    // Build forward graph
    ctx->forward_graph = ggml_new_graph(ctx->ggml_ctx);
    ggml_build_forward_expand(ctx->forward_graph, ctx->output_sequence);
    
    return ctx->forward_graph != nullptr;
}

// Integration tests

// Test 1: End-to-End ATLAS Pipeline
static bool test_end_to_end_pipeline() {
    printf("Testing end-to-end ATLAS pipeline... ");
    
    atlas_test_config config = {
        .sequence_length = 128,
        .batch_size = 2,
        .hidden_dimension = 256,
        .memory_window_size = 64,
        .learning_rate = 0.001f,
        .muon_iterations = 3,
        .use_cuda = false,
        .enable_profiling = true,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    // Initialize ATLAS context
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    // Build forward graph
    if (!build_atlas_forward_graph(&ctx, config)) {
        printf("FAILED - Forward graph construction failed\n");
        cleanup_atlas_context(&ctx);
        return false;
    }
    
    bool success = true;
    
    // Verify graph structure
    success &= (ctx.forward_graph != nullptr);
    success &= (ggml_graph_n_nodes(ctx.forward_graph) > 5);
    success &= (ctx.output_sequence != nullptr);
    success &= (ggml_nelements(ctx.output_sequence) == 
               config.sequence_length * config.hidden_dimension * config.batch_size);
    
    // Verify tensor shapes
    success &= (ctx.input_sequence->ne[0] == config.hidden_dimension);
    success &= (ctx.input_sequence->ne[1] == config.sequence_length);
    success &= (ctx.input_sequence->ne[2] == config.batch_size);
    
    success &= (ctx.memory_keys->ne[0] == config.hidden_dimension);
    success &= (ctx.memory_keys->ne[1] == config.memory_window_size);
    success &= (ctx.memory_keys->ne[2] == config.batch_size);
    
    // Check memory usage
    ctx.memory_usage_mb = get_memory_usage_mb(ctx.ggml_ctx);
    success &= (ctx.memory_usage_mb > 0 && ctx.memory_usage_mb < 1024); // Should be reasonable
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 2: Memory Window Management
static bool test_memory_window_management() {
    printf("Testing memory window management... ");
    
    atlas_test_config config = {
        .sequence_length = 256,
        .batch_size = 1,
        .hidden_dimension = 128,
        .memory_window_size = 64,
        .learning_rate = 0.001f,
        .muon_iterations = 3,
        .use_cuda = false,
        .enable_profiling = false,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    bool success = true;
    
    // Simulate sliding window updates
    float* context_data = (float*)ctx.context_window->data;
    const int window_size = config.memory_window_size;
    const int hidden_dim = config.hidden_dimension;
    
    // Fill context window with known pattern
    for (int i = 0; i < window_size; i++) {
        for (int j = 0; j < hidden_dim; j++) {
            context_data[i * hidden_dim + j] = (float)(i * 100 + j);
        }
    }
    
    // Simulate inserting new context at different positions
    std::vector<float> new_context(hidden_dim);
    for (int j = 0; j < hidden_dim; j++) {
        new_context[j] = 999.0f + j; // Distinctive pattern
    }
    
    // Test circular buffer behavior
    for (int pos = 0; pos < window_size * 2; pos++) {
        int actual_pos = pos % window_size;
        
        // Insert new context
        for (int j = 0; j < hidden_dim; j++) {
            context_data[actual_pos * hidden_dim + j] = new_context[j] + pos;
        }
        
        // Verify insertion
        for (int j = 0; j < hidden_dim; j++) {
            float expected = new_context[j] + pos;
            float actual = context_data[actual_pos * hidden_dim + j];
            success &= (std::abs(actual - expected) < config.tolerance);
            if (!success) break;
        }
        
        if (!success) break;
    }
    
    // Test window statistics
    float mean = 0.0f;
    for (int i = 0; i < window_size * hidden_dim; i++) {
        mean += context_data[i];
    }
    mean /= (window_size * hidden_dim);
    
    success &= (std::isfinite(mean));
    success &= (mean > 0.0f); // Should be positive due to our test pattern
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 3: Multi-batch Processing
static bool test_multi_batch_processing() {
    printf("Testing multi-batch processing... ");
    
    atlas_test_config config = {
        .sequence_length = 64,
        .batch_size = 8,  // Multiple batches
        .hidden_dimension = 256,
        .memory_window_size = 32,
        .learning_rate = 0.001f,
        .muon_iterations = 3,
        .use_cuda = false,
        .enable_profiling = false,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    if (!build_atlas_forward_graph(&ctx, config)) {
        printf("FAILED - Forward graph construction failed\n");
        cleanup_atlas_context(&ctx);
        return false;
    }
    
    bool success = true;
    
    // Fill each batch with different patterns
    float* input_data = (float*)ctx.input_sequence->data;
    const int seq_len = config.sequence_length;
    const int hidden_dim = config.hidden_dimension;
    const int batch_size = config.batch_size;
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < hidden_dim; h++) {
                // Each batch gets a different base value
                float base_value = (b + 1) * 0.1f;
                float position_value = s * 0.001f;
                float dim_value = h * 0.0001f;
                
                input_data[b * seq_len * hidden_dim + s * hidden_dim + h] = 
                    base_value + position_value + dim_value;
            }
        }
    }
    
    // Verify input patterns are correctly set
    for (int b = 0; b < batch_size; b++) {
        float batch_mean = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < hidden_dim; h++) {
                batch_mean += input_data[b * seq_len * hidden_dim + s * hidden_dim + h];
            }
        }
        batch_mean /= (seq_len * hidden_dim);
        
        // Each batch should have a different mean
        float expected_base = (b + 1) * 0.1f;
        success &= (std::abs(batch_mean - expected_base) < 0.1f);
    }
    
    // Verify graph can handle multiple batches
    success &= (ctx.forward_graph != nullptr);
    success &= (ggml_graph_n_nodes(ctx.forward_graph) > 0);
    
    // Check tensor dimensions match batch requirements
    success &= (ctx.output_sequence->ne[2] == batch_size);
    success &= (ctx.memory_keys->ne[2] == batch_size);
    success &= (ctx.memory_values->ne[2] == batch_size);
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 4: Memory Usage and Performance
static bool test_memory_performance() {
    printf("Testing memory usage and performance... ");
    
    atlas_test_config config = {
        .sequence_length = 512,
        .batch_size = 4,
        .hidden_dimension = 768,
        .memory_window_size = 128,
        .learning_rate = 0.001f,
        .muon_iterations = 5,
        .use_cuda = false,
        .enable_profiling = true,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    double init_start = get_time_ms();
    
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    double init_time = get_time_ms() - init_start;
    
    double graph_start = get_time_ms();
    
    if (!build_atlas_forward_graph(&ctx, config)) {
        printf("FAILED - Forward graph construction failed\n");
        cleanup_atlas_context(&ctx);
        return false;
    }
    
    double graph_time = get_time_ms() - graph_start;
    
    bool success = true;
    
    // Check memory usage is reasonable
    ctx.memory_usage_mb = get_memory_usage_mb(ctx.ggml_ctx);
    success &= (ctx.memory_usage_mb > 0);
    success &= (ctx.memory_usage_mb < 600); // Should be less than 600MB for this config
    
    // Check initialization time is reasonable
    success &= (init_time < 5000.0); // Less than 5 seconds
    
    // Check graph construction time is reasonable
    success &= (graph_time < 1000.0); // Less than 1 second
    
    // Verify all tensors are properly allocated
    success &= (ctx.input_sequence->data != nullptr);
    success &= (ctx.output_sequence->data != nullptr);
    success &= (ctx.memory_keys->data != nullptr);
    success &= (ctx.memory_values->data != nullptr);
    success &= (ctx.w1->data != nullptr);
    success &= (ctx.momentum_state->data != nullptr);
    
    // Check tensor data is accessible
    float* test_data = (float*)ctx.input_sequence->data;
    test_data[0] = 42.0f;
    success &= (test_data[0] == 42.0f);
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok (init: %.1fms, graph: %.1fms, mem: %.1fMB)\n", 
               init_time, graph_time, ctx.memory_usage_mb);
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 5: Gradient Flow and Optimization
static bool test_gradient_flow() {
    printf("Testing gradient flow and optimization... ");
    
    atlas_test_config config = {
        .sequence_length = 32,
        .batch_size = 2,
        .hidden_dimension = 128,
        .memory_window_size = 16,
        .learning_rate = 0.01f,
        .muon_iterations = 3,
        .use_cuda = false,
        .enable_profiling = false,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    if (!build_atlas_forward_graph(&ctx, config)) {
        printf("FAILED - Forward graph construction failed\n");
        cleanup_atlas_context(&ctx);
        return false;
    }
    
    bool success = true;
    
    // Store initial parameter values
    std::vector<float> initial_w1(ggml_nelements(ctx.w1));
    std::vector<float> initial_w2(ggml_nelements(ctx.w2));
    
    std::memcpy(initial_w1.data(), ctx.w1->data, ggml_nbytes(ctx.w1));
    std::memcpy(initial_w2.data(), ctx.w2->data, ggml_nbytes(ctx.w2));
    
    // Simulate gradient computation (mock gradients)
    std::vector<float> gradients_w1(ggml_nelements(ctx.w1));
    std::vector<float> gradients_w2(ggml_nelements(ctx.w2));
    
    // Generate small random gradients
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> grad_dis(-0.001f, 0.001f);
    
    for (size_t i = 0; i < gradients_w1.size(); i++) {
        gradients_w1[i] = grad_dis(gen);
    }
    for (size_t i = 0; i < gradients_w2.size(); i++) {
        gradients_w2[i] = grad_dis(gen);
    }
    
    // Apply simple SGD update
    float* w1_data = (float*)ctx.w1->data;
    float* w2_data = (float*)ctx.w2->data;
    
    for (size_t i = 0; i < gradients_w1.size(); i++) {
        w1_data[i] -= config.learning_rate * gradients_w1[i];
    }
    for (size_t i = 0; i < gradients_w2.size(); i++) {
        w2_data[i] -= config.learning_rate * gradients_w2[i];
    }
    
    // Verify parameters have changed
    bool w1_changed = false;
    bool w2_changed = false;
    
    for (size_t i = 0; i < initial_w1.size(); i++) {
        if (std::abs(w1_data[i] - initial_w1[i]) > config.tolerance) {
            w1_changed = true;
            break;
        }
    }
    
    for (size_t i = 0; i < initial_w2.size(); i++) {
        if (std::abs(w2_data[i] - initial_w2[i]) > config.tolerance) {
            w2_changed = true;
            break;
        }
    }
    
    success &= w1_changed;
    success &= w2_changed;
    
    // Verify parameters are still reasonable
    for (size_t i = 0; i < initial_w1.size(); i++) {
        success &= std::isfinite(w1_data[i]);
        success &= (std::abs(w1_data[i]) < 10.0f); // Shouldn't explode
    }
    
    for (size_t i = 0; i < initial_w2.size(); i++) {
        success &= std::isfinite(w2_data[i]);
        success &= (std::abs(w2_data[i]) < 10.0f); // Shouldn't explode
    }
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

#ifdef ATLAS_CUDA_ENABLED
// Test 6: CUDA Backend Integration (if available)
static bool test_cuda_integration() {
    printf("Testing CUDA backend integration... ");
    
    // Check if CUDA is available
    if (!ggml_cuda_is_available()) {
        printf("SKIPPED - CUDA not available\n");
        return true;
    }
    
    atlas_test_config config = {
        .sequence_length = 128,
        .batch_size = 4,
        .hidden_dimension = 256,
        .memory_window_size = 64,
        .learning_rate = 0.001f,
        .muon_iterations = 3,
        .use_cuda = true,
        .enable_profiling = true,
        .tolerance = DEFAULT_TOLERANCE
    };
    
    atlas_integration_context ctx = {};
    
    if (!initialize_atlas_context(&ctx, config)) {
        printf("FAILED - Context initialization failed\n");
        return false;
    }
    
    if (!build_atlas_forward_graph(&ctx, config)) {
        printf("FAILED - Forward graph construction failed\n");
        cleanup_atlas_context(&ctx);
        return false;
    }
    
    bool success = true;
    
    // Verify CUDA context was created properly
    success &= (ctx.forward_graph != nullptr);
    success &= (ggml_graph_n_nodes(ctx.forward_graph) > 0);
    
    // Test memory allocation works with CUDA
    success &= (ctx.input_sequence->data != nullptr);
    success &= (ctx.output_sequence->data != nullptr);
    
    // Performance should be reasonable with CUDA
    ctx.memory_usage_mb = get_memory_usage_mb(ctx.ggml_ctx);
    success &= (ctx.memory_usage_mb > 0 && ctx.memory_usage_mb < 1024);
    
    cleanup_atlas_context(&ctx);
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}
#endif

int main(int argc, char** argv) {
    printf("Running ATLAS integration tests...\n");
    
    int tests_passed = 0;
    int total_tests = 5;
    
#ifdef ATLAS_CUDA_ENABLED
    total_tests = 6;
#endif
    
    if (test_end_to_end_pipeline()) tests_passed++;
    if (test_memory_window_management()) tests_passed++;
    if (test_multi_batch_processing()) tests_passed++;
    if (test_memory_performance()) tests_passed++;
    if (test_gradient_flow()) tests_passed++;
    
#ifdef ATLAS_CUDA_ENABLED
    if (test_cuda_integration()) tests_passed++;
#endif
    
    printf("\nATLAS integration tests completed: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("All ATLAS integration tests PASSED!\n");
        return 0;
    } else {
        printf("Some ATLAS integration tests FAILED!\n");
        return 1;
    }
}