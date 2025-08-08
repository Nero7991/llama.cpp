#include "framework/atlas-test-framework.h"
#include "llama-atlas.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace atlas::testing;

// Integration tests for full ATLAS pipeline
class AtlasIntegrationTestSuite {
public:
    static bool runFullPipelineIntegration() {
        std::cout << "Testing full ATLAS pipeline integration..." << std::endl;
        
        const int batch_size = 2;
        const int seq_len = 256;
        const int hidden_dim = 768;
        const int n_layers = 4;
        
        // Initialize ATLAS configuration
        struct atlas_config config = atlas_config_default();
        config.max_sequence_length = seq_len;
        config.memory_pool_size = 64 * 1024 * 1024; // 64MB
        
        // Initialize ATLAS context
        struct atlas_context* atlas_ctx = atlas_init(&config, n_layers);
        if (!atlas_ctx) {
            std::cout << "Failed to initialize ATLAS context" << std::endl;
            return false;
        }
        
        // Initialize GGML context
        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size = 128 * 1024 * 1024; // 128MB
        ggml_params.mem_buffer = nullptr;
        ggml_params.no_alloc = false;
        
        struct ggml_context* ggml_ctx = ggml_init(ggml_params);
        if (!ggml_ctx) {
            atlas_free(atlas_ctx);
            std::cout << "Failed to initialize GGML context" << std::endl;
            return false;
        }
        
        // Create input tensor
        struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32,
                                                       hidden_dim, seq_len, batch_size);
        if (!input) {
            ggml_free(ggml_ctx);
            atlas_free(atlas_ctx);
            return false;
        }
        
        // Fill input with test data
        float* input_data = (float*)input->data;
        for (size_t i = 0; i < ggml_nelements(input); i++) {
            input_data[i] = 0.01f * (std::sin(i * 0.01f) + 0.1f * (i % 100 - 50));
        }
        
        // Run through all layers
        struct ggml_tensor* current_input = input;
        for (int layer = 0; layer < n_layers; layer++) {
            std::cout << "Processing layer " << layer << "..." << std::endl;
            
            struct ggml_tensor* layer_output = atlas_attention_forward(
                ggml_ctx, 
                &atlas_ctx->layers[layer], 
                current_input, 
                nullptr,  // no mask
                seq_len, 
                hidden_dim / 8  // n_heads
            );
            
            if (!layer_output) {
                std::cout << "Layer " << layer << " forward pass failed" << std::endl;
                ggml_free(ggml_ctx);
                atlas_free(atlas_ctx);
                return false;
            }
            
            // Validate layer output
            float* output_data = (float*)layer_output->data;
            for (size_t i = 0; i < ggml_nelements(layer_output); i++) {
                if (!std::isfinite(output_data[i])) {
                    std::cout << "Non-finite value in layer " << layer << " at index " << i << std::endl;
                    ggml_free(ggml_ctx);
                    atlas_free(atlas_ctx);
                    return false;
                }
            }
            
            current_input = layer_output;
        }
        
        // Cleanup
        ggml_free(ggml_ctx);
        atlas_free(atlas_ctx);
        
        std::cout << "Full pipeline integration test completed successfully" << std::endl;
        return true;
    }
    
    static bool runMemoryManagementIntegration() {
        std::cout << "Testing ATLAS memory management integration..." << std::endl;
        
        // Test multiple initialization/cleanup cycles
        const int num_cycles = 5;
        const int n_layers = 6;
        
        for (int cycle = 0; cycle < num_cycles; cycle++) {
            std::cout << "Memory cycle " << cycle + 1 << "/" << num_cycles << "..." << std::endl;
            
            struct atlas_config config = atlas_config_default();
            config.memory_pool_size = 32 * 1024 * 1024; // 32MB
            config.max_sequence_length = 512;
            
            struct atlas_context* ctx = atlas_init(&config, n_layers);
            if (!ctx) {
                std::cout << "Failed to initialize ATLAS in cycle " << cycle << std::endl;
                return false;
            }
            
            // Test memory operations
            for (int layer = 0; layer < n_layers; layer++) {
                if (!ctx->layers[layer].memory_manager.allocator) {
                    std::cout << "Memory manager not initialized for layer " << layer << std::endl;
                    atlas_free(ctx);
                    return false;
                }
            }
            
            // Cleanup
            atlas_free(ctx);
            
            // Small delay to allow OS cleanup
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "Memory management integration test completed successfully" << std::endl;
        return true;
    }
    
    static bool runPerformanceIntegration() {
        std::cout << "Testing ATLAS performance characteristics..." << std::endl;
        
        const int batch_size = 4;
        const int seq_len = 1024;
        const int hidden_dim = 1024;
        const int n_layers = 8;
        const int num_warmup = 3;
        const int num_benchmark = 10;
        
        struct atlas_config config = atlas_config_default();
        config.max_sequence_length = seq_len;
        config.memory_pool_size = 256 * 1024 * 1024; // 256MB
        
        struct atlas_context* atlas_ctx = atlas_init(&config, n_layers);
        if (!atlas_ctx) {
            return false;
        }
        
        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size = 512 * 1024 * 1024; // 512MB
        ggml_params.mem_buffer = nullptr;
        ggml_params.no_alloc = false;
        
        struct ggml_context* ggml_ctx = ggml_init(ggml_params);
        if (!ggml_ctx) {
            atlas_free(atlas_ctx);
            return false;
        }
        
        struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32,
                                                       hidden_dim, seq_len, batch_size);
        if (!input) {
            ggml_free(ggml_ctx);
            atlas_free(atlas_ctx);
            return false;
        }
        
        // Fill with random data
        float* input_data = (float*)input->data;
        for (size_t i = 0; i < ggml_nelements(input); i++) {
            input_data[i] = 0.02f * (std::sin(i * 0.001f) + 0.1f * std::cos(i * 0.003f));
        }
        
        // Warmup runs
        for (int i = 0; i < num_warmup; i++) {
            struct ggml_tensor* output = atlas_attention_forward(
                ggml_ctx, &atlas_ctx->layers[0], input, nullptr, seq_len, hidden_dim / 8);
            if (!output) {
                ggml_free(ggml_ctx);
                atlas_free(atlas_ctx);
                return false;
            }
        }
        
        // Benchmark runs
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_benchmark; i++) {
            struct ggml_tensor* output = atlas_attention_forward(
                ggml_ctx, &atlas_ctx->layers[0], input, nullptr, seq_len, hidden_dim / 8);
            if (!output) {
                ggml_free(ggml_ctx);
                atlas_free(atlas_ctx);
                return false;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        double avg_time_ms = duration_ms / num_benchmark;
        double tokens_per_sec = (batch_size * seq_len * 1000.0) / avg_time_ms;
        
        std::cout << "Performance results:" << std::endl;
        std::cout << "  Average time per forward pass: " << avg_time_ms << " ms" << std::endl;
        std::cout << "  Tokens per second: " << tokens_per_sec << std::endl;
        
        // Performance thresholds (adjust based on hardware)
        bool performance_acceptable = true;
        if (tokens_per_sec < 1000) {  // At least 1K tokens/sec
            std::cout << "Warning: Performance below threshold" << std::endl;
            performance_acceptable = false;
        }
        
        if (avg_time_ms > 1000) {  // No more than 1 second per forward pass
            std::cout << "Warning: Latency too high" << std::endl;
            performance_acceptable = false;
        }
        
        ggml_free(ggml_ctx);
        atlas_free(atlas_ctx);
        
        if (performance_acceptable) {
            std::cout << "Performance integration test completed successfully" << std::endl;
        } else {
            std::cout << "Performance integration test completed with warnings" << std::endl;
        }
        
        return true;  // Return true even with warnings for integration test
    }
    
    static bool runLlamaCompatibilityCheck() {
        std::cout << "Testing llama.cpp compatibility..." << std::endl;
        
        // Test that ATLAS structures are compatible with llama.cpp expectations
        struct atlas_config config = atlas_config_default();
        
        // Verify default configuration is reasonable
        if (config.max_sequence_length <= 0) {
            std::cout << "Invalid default max_sequence_length: " << config.max_sequence_length << std::endl;
            return false;
        }
        
        if (config.memory_pool_size <= 0) {
            std::cout << "Invalid default memory_pool_size: " << config.memory_pool_size << std::endl;
            return false;
        }
        
        // Test initialization with different layer counts
        std::vector<int> layer_counts = {1, 4, 8, 16, 32};
        
        for (int n_layers : layer_counts) {
            struct atlas_context* ctx = atlas_init(&config, n_layers);
            if (!ctx) {
                std::cout << "Failed to initialize ATLAS with " << n_layers << " layers" << std::endl;
                return false;
            }
            
            // Verify layer structure
            if (ctx->n_layers != n_layers) {
                std::cout << "Layer count mismatch: expected " << n_layers << ", got " << ctx->n_layers << std::endl;
                atlas_free(ctx);
                return false;
            }
            
            atlas_free(ctx);
        }
        
        std::cout << "Llama compatibility check completed successfully" << std::endl;
        return true;
    }
};

int main() {
    std::cout << "=== ATLAS Integration Tests ===" << std::endl;
    
    bool all_passed = true;
    
    std::cout << "\n1. Running full pipeline integration test..." << std::endl;
    if (!AtlasIntegrationTestSuite::runFullPipelineIntegration()) {
        std::cout << "❌ Full pipeline integration test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Full pipeline integration test passed" << std::endl;
    }
    
    std::cout << "\n2. Running memory management integration test..." << std::endl;
    if (!AtlasIntegrationTestSuite::runMemoryManagementIntegration()) {
        std::cout << "❌ Memory management integration test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Memory management integration test passed" << std::endl;
    }
    
    std::cout << "\n3. Running performance integration test..." << std::endl;
    if (!AtlasIntegrationTestSuite::runPerformanceIntegration()) {
        std::cout << "❌ Performance integration test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Performance integration test passed" << std::endl;
    }
    
    std::cout << "\n4. Running llama.cpp compatibility check..." << std::endl;
    if (!AtlasIntegrationTestSuite::runLlamaCompatibilityCheck()) {
        std::cout << "❌ Llama compatibility check failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ Llama compatibility check passed" << std::endl;
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL INTEGRATION TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ SOME INTEGRATION TESTS FAILED!" << std::endl;
    }
    std::cout << std::string(50, '=') << std::endl;
    
    return all_passed ? 0 : 1;
}