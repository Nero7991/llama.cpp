#include "framework/atlas-test-framework.h"
#include "llama-atlas.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fstream>

using namespace atlas::testing;

// Comprehensive benchmarking for ATLAS components
class AtlasBenchmarkSuite {
public:
    static void runMemoryModuleBenchmark() {
        std::cout << "=== Memory Module Benchmark ===" << std::endl;
        
        std::vector<int> batch_sizes = {1, 2, 4, 8, 16};
        std::vector<int> seq_lengths = {128, 256, 512, 1024, 2048};
        std::vector<int> hidden_dims = {512, 768, 1024, 1536, 2048};
        
        const int num_runs = 50;
        const int warmup_runs = 10;
        
        struct atlas_config config = atlas_config_default();
        config.memory_pool_size = 512 * 1024 * 1024; // 512MB
        
        std::cout << "Configuration,Batch,SeqLen,HiddenDim,AvgTime(ms),StdDev(ms),TokensPerSec,GFLOPS" << std::endl;
        
        for (int batch_size : batch_sizes) {
            for (int seq_len : seq_lengths) {
                for (int hidden_dim : hidden_dims) {
                    config.max_sequence_length = seq_len;
                    
                    struct atlas_context* atlas_ctx = atlas_init(&config, 1);
                    if (!atlas_ctx) continue;
                    
                    struct ggml_init_params ggml_params = {};
                    ggml_params.mem_size = (size_t)(batch_size * seq_len * hidden_dim) * 2 * sizeof(float) + 64 * 1024;
                    ggml_params.mem_buffer = nullptr;
                    ggml_params.no_alloc = false;
                    
                    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
                    if (!ggml_ctx) {
                        atlas_free(atlas_ctx);
                        continue;
                    }
                    
                    struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32,
                                                                   hidden_dim, seq_len, batch_size);
                    if (!input) {
                        ggml_free(ggml_ctx);
                        atlas_free(atlas_ctx);
                        continue;
                    }
                    
                    // Fill with test data
                    float* input_data = (float*)input->data;
                    for (size_t i = 0; i < ggml_nelements(input); i++) {
                        input_data[i] = 0.01f * std::sin(i * 0.001f);
                    }
                    
                    // Warmup
                    for (int i = 0; i < warmup_runs; i++) {
                        atlas_attention_forward(ggml_ctx, &atlas_ctx->layers[0], input, nullptr, seq_len, 8);
                    }
                    
                    // Benchmark
                    std::vector<double> times;
                    times.reserve(num_runs);
                    
                    for (int i = 0; i < num_runs; i++) {
                        auto start = std::chrono::high_resolution_clock::now();
                        
                        struct ggml_tensor* output = atlas_attention_forward(
                            ggml_ctx, &atlas_ctx->layers[0], input, nullptr, seq_len, 8);
                        
                        auto end = std::chrono::high_resolution_clock::now();
                        
                        if (output) {
                            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                            times.push_back(time_ms);
                        }
                    }
                    
                    if (!times.empty()) {
                        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                        
                        double variance = 0.0;
                        for (double time : times) {
                            variance += (time - avg_time) * (time - avg_time);
                        }
                        double std_dev = std::sqrt(variance / times.size());
                        
                        double tokens_per_sec = (batch_size * seq_len * 1000.0) / avg_time;
                        
                        // Estimate GFLOPS for 2-layer MLP with residual connections
                        double ops_per_token = hidden_dim * hidden_dim * 4; // Two linear layers
                        double gflops = (batch_size * seq_len * ops_per_token) / (avg_time / 1000.0) / 1e9;
                        
                        std::cout << "MemoryModule," << batch_size << "," << seq_len << "," << hidden_dim << ","
                                  << std::fixed << std::setprecision(2) << avg_time << ","
                                  << std::setprecision(2) << std_dev << ","
                                  << std::setprecision(0) << tokens_per_sec << ","
                                  << std::setprecision(1) << gflops << std::endl;
                    }
                    
                    ggml_free(ggml_ctx);
                    atlas_free(atlas_ctx);
                }
            }
        }
    }
    
    static void runScalabilityBenchmark() {
        std::cout << "\n=== Scalability Benchmark ===" << std::endl;
        
        std::vector<int> layer_counts = {1, 2, 4, 8, 12, 16, 24, 32};
        const int batch_size = 4;
        const int seq_len = 512;
        const int hidden_dim = 1024;
        const int num_runs = 20;
        
        std::cout << "Layers,Time(ms),TokensPerSec,MemoryMB,TimePerLayer(ms)" << std::endl;
        
        for (int n_layers : layer_counts) {
            struct atlas_config config = atlas_config_default();
            config.max_sequence_length = seq_len;
            config.memory_pool_size = n_layers * 64 * 1024 * 1024; // 64MB per layer
            
            struct atlas_context* atlas_ctx = atlas_init(&config, n_layers);
            if (!atlas_ctx) continue;
            
            struct ggml_init_params ggml_params = {};
            ggml_params.mem_size = (size_t)(batch_size * seq_len * hidden_dim * n_layers) * 2 * sizeof(float) + 1024 * 1024;
            ggml_params.mem_buffer = nullptr;
            ggml_params.no_alloc = false;
            
            struct ggml_context* ggml_ctx = ggml_init(ggml_params);
            if (!ggml_ctx) {
                atlas_free(atlas_ctx);
                continue;
            }
            
            struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32,
                                                           hidden_dim, seq_len, batch_size);
            if (!input) {
                ggml_free(ggml_ctx);
                atlas_free(atlas_ctx);
                continue;
            }
            
            // Fill with test data
            float* input_data = (float*)input->data;
            for (size_t i = 0; i < ggml_nelements(input); i++) {
                input_data[i] = 0.01f * std::tanh(i * 0.0001f);
            }
            
            std::vector<double> times;
            times.reserve(num_runs);
            
            // Benchmark full pipeline
            for (int run = 0; run < num_runs; run++) {
                auto start = std::chrono::high_resolution_clock::now();
                
                struct ggml_tensor* current_input = input;
                
                for (int layer = 0; layer < n_layers; layer++) {
                    struct ggml_tensor* layer_output = atlas_attention_forward(
                        ggml_ctx, &atlas_ctx->layers[layer], current_input, nullptr, seq_len, 8);
                    
                    if (!layer_output) break;
                    current_input = layer_output;
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                times.push_back(time_ms);
            }
            
            if (!times.empty()) {
                double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
                double tokens_per_sec = (batch_size * seq_len * 1000.0) / avg_time;
                double memory_mb = config.memory_pool_size / (1024.0 * 1024.0);
                double time_per_layer = avg_time / n_layers;
                
                std::cout << n_layers << "," << std::fixed << std::setprecision(2) << avg_time << ","
                          << std::setprecision(0) << tokens_per_sec << ","
                          << std::setprecision(1) << memory_mb << ","
                          << std::setprecision(2) << time_per_layer << std::endl;
            }
            
            ggml_free(ggml_ctx);
            atlas_free(atlas_ctx);
        }
    }
    
    static void runMemoryBandwidthBenchmark() {
        std::cout << "\n=== Memory Bandwidth Benchmark ===" << std::endl;
        
        std::vector<int> data_sizes_mb = {1, 2, 4, 8, 16, 32, 64, 128};
        const int num_runs = 100;
        
        std::cout << "DataSize(MB),ReadBW(GB/s),WriteBW(GB/s),CopyBW(GB/s)" << std::endl;
        
        for (int size_mb : data_sizes_mb) {
            size_t size_bytes = size_mb * 1024 * 1024;
            size_t num_floats = size_bytes / sizeof(float);
            
            std::vector<float> src(num_floats);
            std::vector<float> dst(num_floats);
            
            // Fill source with test data
            for (size_t i = 0; i < num_floats; i++) {
                src[i] = 0.1f * std::sin(i * 0.001f);
            }
            
            // Read benchmark
            volatile float sum = 0.0f;
            auto start = std::chrono::high_resolution_clock::now();
            
            for (int run = 0; run < num_runs; run++) {
                for (size_t i = 0; i < num_floats; i += 4) { // Unroll for better performance
                    sum += src[i] + src[i+1] + src[i+2] + src[i+3];
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double read_time = std::chrono::duration<double>(end - start).count();
            double read_bw = (size_bytes * num_runs) / read_time / 1e9;
            
            // Write benchmark
            start = std::chrono::high_resolution_clock::now();
            
            for (int run = 0; run < num_runs; run++) {
                for (size_t i = 0; i < num_floats; i += 4) { // Unroll for better performance
                    dst[i] = 0.1f;
                    dst[i+1] = 0.2f;
                    dst[i+2] = 0.3f;
                    dst[i+3] = 0.4f;
                }
            }
            
            end = std::chrono::high_resolution_clock::now();
            double write_time = std::chrono::duration<double>(end - start).count();
            double write_bw = (size_bytes * num_runs) / write_time / 1e9;
            
            // Copy benchmark
            start = std::chrono::high_resolution_clock::now();
            
            for (int run = 0; run < num_runs; run++) {
                std::memcpy(dst.data(), src.data(), size_bytes);
            }
            
            end = std::chrono::high_resolution_clock::now();
            double copy_time = std::chrono::duration<double>(end - start).count();
            double copy_bw = (size_bytes * num_runs) / copy_time / 1e9;
            
            std::cout << size_mb << "," << std::fixed << std::setprecision(2) 
                      << read_bw << "," << write_bw << "," << copy_bw << std::endl;
        }
    }
    
    static void runLatencyAnalysis() {
        std::cout << "\n=== Latency Analysis ===" << std::endl;
        
        const int batch_size = 1; // Single batch for latency measurement
        const int seq_len = 1024;
        const int hidden_dim = 1024;
        const int num_runs = 1000;
        
        struct atlas_config config = atlas_config_default();
        config.max_sequence_length = seq_len;
        config.memory_pool_size = 128 * 1024 * 1024; // 128MB
        
        struct atlas_context* atlas_ctx = atlas_init(&config, 1);
        if (!atlas_ctx) return;
        
        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size = (size_t)(batch_size * seq_len * hidden_dim) * 4 * sizeof(float);
        ggml_params.mem_buffer = nullptr;
        ggml_params.no_alloc = false;
        
        struct ggml_context* ggml_ctx = ggml_init(ggml_params);
        if (!ggml_ctx) {
            atlas_free(atlas_ctx);
            return;
        }
        
        struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32,
                                                       hidden_dim, seq_len, batch_size);
        if (!input) {
            ggml_free(ggml_ctx);
            atlas_free(atlas_ctx);
            return;
        }
        
        // Fill with test data
        float* input_data = (float*)input->data;
        for (size_t i = 0; i < ggml_nelements(input); i++) {
            input_data[i] = 0.01f * std::sin(i * 0.001f);
        }
        
        std::vector<double> latencies;
        latencies.reserve(num_runs);
        
        // Measure individual run latencies
        for (int run = 0; run < num_runs; run++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            struct ggml_tensor* output = atlas_attention_forward(
                ggml_ctx, &atlas_ctx->layers[0], input, nullptr, seq_len, 8);
            
            auto end = std::chrono::high_resolution_clock::now();
            
            if (output) {
                double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
                latencies.push_back(latency_ms);
            }
        }
        
        if (!latencies.empty()) {
            std::sort(latencies.begin(), latencies.end());
            
            double p50 = latencies[latencies.size() * 50 / 100];
            double p90 = latencies[latencies.size() * 90 / 100];
            double p95 = latencies[latencies.size() * 95 / 100];
            double p99 = latencies[latencies.size() * 99 / 100];
            double min_lat = *std::min_element(latencies.begin(), latencies.end());
            double max_lat = *std::max_element(latencies.begin(), latencies.end());
            double avg_lat = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
            
            std::cout << "Latency Statistics (ms):" << std::endl;
            std::cout << "  Min: " << std::fixed << std::setprecision(2) << min_lat << std::endl;
            std::cout << "  Avg: " << std::setprecision(2) << avg_lat << std::endl;
            std::cout << "  P50: " << std::setprecision(2) << p50 << std::endl;
            std::cout << "  P90: " << std::setprecision(2) << p90 << std::endl;
            std::cout << "  P95: " << std::setprecision(2) << p95 << std::endl;
            std::cout << "  P99: " << std::setprecision(2) << p99 << std::endl;
            std::cout << "  Max: " << std::setprecision(2) << max_lat << std::endl;
        }
        
        ggml_free(ggml_ctx);
        atlas_free(atlas_ctx);
    }
};

int main() {
    std::cout << "=== ATLAS Comprehensive Benchmark Suite ===" << std::endl;
    std::cout << "Timestamp: " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
    
    AtlasBenchmarkSuite::runMemoryModuleBenchmark();
    AtlasBenchmarkSuite::runScalabilityBenchmark();
    AtlasBenchmarkSuite::runMemoryBandwidthBenchmark();
    AtlasBenchmarkSuite::runLatencyAnalysis();
    
    std::cout << "\n=== Benchmark Complete ===" << std::endl;
    
    return 0;
}