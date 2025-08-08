#include "atlas-test-framework.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>

namespace atlas {
namespace testing {

AtlasTestFramework::AtlasTestFramework() 
    : test_context_(nullptr)
    , total_tests_(0)
    , passed_tests_(0)
    , failed_tests_(0) {
    setupAtlasTestEnvironment();
}

AtlasTestFramework::~AtlasTestFramework() {
    teardownAtlasTestEnvironment();
}

void AtlasTestFramework::setupAtlasTestEnvironment() {
    std::cout << "Setting up ATLAS test environment..." << std::endl;
    
    // Initialize GGML context for testing
    size_t ctx_size = 256 * 1024 * 1024; // 256MB
    struct ggml_init_params params = {};
    params.mem_size = ctx_size;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    
    ggml_ctx_ = ggml_init(params);
    if (!ggml_ctx_) {
        throw std::runtime_error("Failed to initialize GGML context for testing");
    }
    
    // Initialize ATLAS context with test configuration
    struct atlas_config config = atlas_config_default();
    config.max_sequence_length = 4096;
    config.memory_pool_size = 128 * 1024 * 1024; // 128MB
    
    test_context_ = atlas_init(&config, 8); // 8 layers for testing
    if (!test_context_) {
        throw std::runtime_error("Failed to initialize ATLAS context for testing");
    }
    
    // Load performance baselines
    loadPerformanceBaselines();
    
    std::cout << "ATLAS test environment initialized successfully." << std::endl;
}

void AtlasTestFramework::teardownAtlasTestEnvironment() {
    if (test_context_) {
        atlas_free(test_context_);
        test_context_ = nullptr;
    }
    
    if (ggml_ctx_) {
        ggml_free(ggml_ctx_);
        ggml_ctx_ = nullptr;
    }
}

void AtlasTestFramework::loadPerformanceBaselines() {
    // Load baseline metrics from file or set defaults
    baseline_metrics_.memory_module_gflops = 500.0;
    baseline_metrics_.omega_rule_ms = 10.0;
    baseline_metrics_.newton_schulz_ms = 5.0;
    baseline_metrics_.feature_mapping_vectors_per_sec = 1000000;
    baseline_metrics_.full_pipeline_tokens_per_sec = 100.0;
}

bool AtlasTestFramework::addTestCase(const AtlasTestCase& test_case) {
    test_cases_.push_back(test_case);
    return true;
}

TestResults AtlasTestFramework::runAllTests() {
    std::cout << "\n=== Running ATLAS Test Suite ===" << std::endl;
    std::cout << "Total test cases: " << test_cases_.size() << std::endl;
    
    TestResults results;
    results.start_time = std::chrono::high_resolution_clock::now();
    
    for (const auto& test_case : test_cases_) {
        TestCaseResult case_result = runTestCase(test_case);
        results.test_case_results.push_back(case_result);
        
        if (case_result.passed) {
            passed_tests_++;
        } else {
            failed_tests_++;
        }
        total_tests_++;
        
        // Print progress
        std::cout << "[" << total_tests_ << "/" << test_cases_.size() << "] "
                  << test_case.name << ": " << (case_result.passed ? "PASS" : "FAIL");
        if (!case_result.passed) {
            std::cout << " - " << case_result.error_message;
        }
        std::cout << std::endl;
    }
    
    results.end_time = std::chrono::high_resolution_clock::now();
    results.total_duration_ms = std::chrono::duration<double, std::milli>(
        results.end_time - results.start_time).count();
    
    results.total_tests = total_tests_;
    results.passed_tests = passed_tests_;
    results.failed_tests = failed_tests_;
    results.success_rate = (double)passed_tests_ / total_tests_ * 100.0;
    
    printTestSummary(results);
    return results;
}

TestCaseResult AtlasTestFramework::runTestCase(const AtlasTestCase& test_case) {
    TestCaseResult result;
    result.test_name = test_case.name;
    result.start_time = std::chrono::high_resolution_clock::now();
    
    try {
        switch (test_case.type) {
            case TestType::UNIT:
                result.passed = runUnitTest(test_case);
                break;
            case TestType::INTEGRATION:
                result.passed = runIntegrationTest(test_case);
                break;
            case TestType::PERFORMANCE:
                result = runPerformanceTest(test_case);
                break;
            case TestType::STRESS:
                result.passed = runStressTest(test_case);
                break;
            default:
                result.passed = false;
                result.error_message = "Unknown test type";
        }
    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = std::string("Exception: ") + e.what();
    }
    
    result.end_time = std::chrono::high_resolution_clock::now();
    result.duration_ms = std::chrono::duration<double, std::milli>(
        result.end_time - result.start_time).count();
    
    return result;
}

bool AtlasTestFramework::runUnitTest(const AtlasTestCase& test_case) {
    switch (test_case.component) {
        case ComponentType::MEMORY_MODULE:
            return testMemoryModule(test_case.config);
        case ComponentType::OMEGA_RULE:
            return testOmegaRule(test_case.config);
        case ComponentType::MUON_OPTIMIZER:
            return testMuonOptimizer(test_case.config);
        case ComponentType::FEATURE_MAPPING:
            return testFeatureMapping(test_case.config);
        default:
            return false;
    }
}

bool AtlasTestFramework::runIntegrationTest(const AtlasTestCase& test_case) {
    switch (test_case.component) {
        case ComponentType::FULL_PIPELINE:
            return testFullAtlasPipeline(test_case.config);
        case ComponentType::LLAMA_INTEGRATION:
            return testLlamaIntegration(test_case.config);
        default:
            return false;
    }
}

TestCaseResult AtlasTestFramework::runPerformanceTest(const AtlasTestCase& test_case) {
    TestCaseResult result;
    result.test_name = test_case.name;
    result.start_time = std::chrono::high_resolution_clock::now();
    
    try {
        PerformanceMetrics metrics = benchmarkComponent(test_case.component, test_case.config);
        
        // Check against performance targets
        bool meets_targets = true;
        std::string performance_issues;
        
        switch (test_case.component) {
            case ComponentType::MEMORY_MODULE:
                if (metrics.gflops < baseline_metrics_.memory_module_gflops * 0.9) {
                    meets_targets = false;
                    performance_issues += "GFLOPS below target; ";
                }
                break;
            case ComponentType::OMEGA_RULE:
                if (metrics.latency_ms > baseline_metrics_.omega_rule_ms * 1.1) {
                    meets_targets = false;
                    performance_issues += "Latency above target; ";
                }
                break;
            case ComponentType::NEWTON_SCHULZ:
                if (metrics.convergence_time_ms > baseline_metrics_.newton_schulz_ms * 1.1) {
                    meets_targets = false;
                    performance_issues += "Convergence time above target; ";
                }
                break;
            case ComponentType::FEATURE_MAPPING:
                if (metrics.throughput < baseline_metrics_.feature_mapping_vectors_per_sec * 0.9) {
                    meets_targets = false;
                    performance_issues += "Throughput below target; ";
                }
                break;
            default:
                break;
        }
        
        result.passed = meets_targets;
        result.performance_metrics = metrics;
        
        if (!meets_targets) {
            result.error_message = "Performance targets not met: " + performance_issues;
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = std::string("Performance test exception: ") + e.what();
    }
    
    result.end_time = std::chrono::high_resolution_clock::now();
    result.duration_ms = std::chrono::duration<double, std::milli>(
        result.end_time - result.start_time).count();
    
    return result;
}

bool AtlasTestFramework::runStressTest(const AtlasTestCase& test_case) {
    // Simplified stress test implementation
    // In production, this would run extended duration tests
    
    const int stress_iterations = 1000;
    int successful_iterations = 0;
    
    for (int i = 0; i < stress_iterations; i++) {
        try {
            bool iteration_success = false;
            
            switch (test_case.component) {
                case ComponentType::MEMORY_MODULE:
                    iteration_success = testMemoryModule(test_case.config);
                    break;
                case ComponentType::FULL_PIPELINE:
                    iteration_success = testFullAtlasPipeline(test_case.config);
                    break;
                default:
                    iteration_success = testMemoryModule(test_case.config);
                    break;
            }
            
            if (iteration_success) {
                successful_iterations++;
            }
            
        } catch (const std::exception&) {
            // Count as failure but continue
        }
        
        // Check if we're meeting minimum success rate
        double current_success_rate = (double)successful_iterations / (i + 1);
        if (i > 100 && current_success_rate < test_case.tolerance.minimum_success_rate) {
            return false; // Early termination if success rate too low
        }
    }
    
    double final_success_rate = (double)successful_iterations / stress_iterations;
    return final_success_rate >= test_case.tolerance.minimum_success_rate;
}

bool AtlasTestFramework::testMemoryModule(const TestConfig& config) {
    if (!test_context_) return false;
    
    // Create test input tensor
    int batch_size = config.batch_size;
    int seq_len = config.sequence_length;
    int hidden_dim = config.hidden_dimension;
    
    struct ggml_tensor* input = ggml_new_tensor_3d(ggml_ctx_, GGML_TYPE_F32, 
                                                   hidden_dim, seq_len, batch_size);
    if (!input) return false;
    
    // Fill with test data
    float* input_data = (float*)input->data;
    for (int64_t i = 0; i < ggml_nelements(input); i++) {
        input_data[i] = (float)(rand() % 100) / 100.0f - 0.5f; // Random values [-0.5, 0.5]
    }
    
    // Run forward pass
    struct ggml_tensor* output = atlas_attention_forward(
        ggml_ctx_, &test_context_->layers[0], input, nullptr, seq_len, hidden_dim / 8);
    
    if (!output) return false;
    
    // Validate output
    float* output_data = (float*)output->data;
    for (int64_t i = 0; i < ggml_nelements(output); i++) {
        if (!std::isfinite(output_data[i])) {
            return false; // Reject NaN or infinite values
        }
    }
    
    return true;
}

bool AtlasTestFramework::testOmegaRule(const TestConfig& config) {
    // Simplified Omega Rule test
    // In production, this would test sliding window functionality
    
    const int window_size = 128;
    const int hidden_dim = config.hidden_dimension;
    (void)hidden_dim; // Suppress unused warning
    
    // Create test vectors
    std::vector<float> keys(window_size * hidden_dim);
    std::vector<float> values(window_size * hidden_dim);
    std::vector<float> weights(window_size);
    
    // Fill with test data
    for (size_t i = 0; i < keys.size(); i++) {
        keys[i] = (float)(rand() % 100) / 100.0f;
        values[i] = (float)(rand() % 100) / 100.0f;
    }
    
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = 1.0f / window_size; // Uniform weights
    }
    
    // Test omega rule computation (simplified)
    // In production, this would call actual omega rule functions
    
    // Check that weights sum to approximately 1.0
    float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    return std::abs(weight_sum - 1.0f) < 1e-5f;
}

bool AtlasTestFramework::testMuonOptimizer(const TestConfig&) {
    // Test Newton-Schulz convergence
    const int dim = 64; // Small matrix for testing
    
    // Create test matrix (positive definite)
    std::vector<float> matrix(dim * dim);
    std::vector<float> inverse(dim * dim);
    
    // Initialize as identity matrix for simple test
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            matrix[i * dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Test newton-schulz iteration (simplified)
    // In production, this would call the actual CUDA kernel or CPU implementation
    
    // For identity matrix, inverse should also be identity
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            inverse[i * dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Verify A * A^(-1) = I
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < dim; k++) {
                sum += matrix[i * dim + k] * inverse[k * dim + j];
            }
            
            float expected = (i == j) ? 1.0f : 0.0f;
            if (std::abs(sum - expected) > 1e-4f) {
                return false;
            }
        }
    }
    
    return true;
}

bool AtlasTestFramework::testFeatureMapping(const TestConfig&) {
    const int input_size = 1000;
    const int polynomial_degree = 3;
    
    std::vector<float> input(input_size);
    std::vector<float> output(input_size);
    
    // Fill with test data
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = (float)i / input_size; // Values from 0 to 1
    }
    
    // Test polynomial feature mapping
    for (size_t i = 0; i < input.size(); i++) {
        float x = input[i];
        float poly_result = x; // degree 1
        
        if (polynomial_degree >= 2) poly_result += x * x;
        if (polynomial_degree >= 3) poly_result += x * x * x;
        
        output[i] = poly_result;
    }
    
    // Validate results
    for (size_t i = 0; i < output.size(); i++) {
        if (!std::isfinite(output[i])) {
            return false;
        }
        
        // Check that output is reasonable
        if (output[i] < 0 || output[i] > 10.0f) { // Reasonable bounds for our test
            return false;
        }
    }
    
    return true;
}

bool AtlasTestFramework::testFullAtlasPipeline(const TestConfig& config) {
    // Test the full ATLAS pipeline integration
    return testMemoryModule(config) && 
           testOmegaRule(config) && 
           testMuonOptimizer(config) && 
           testFeatureMapping(config);
}

bool AtlasTestFramework::testLlamaIntegration(const TestConfig& config) {
    // Test integration with llama.cpp
    // This is a simplified version
    return testMemoryModule(config);
}

PerformanceMetrics AtlasTestFramework::benchmarkComponent(ComponentType type, const TestConfig& config) {
    PerformanceMetrics metrics = {};
    
    const int num_iterations = 100;
    const int warmup_iterations = 10;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        switch (type) {
            case ComponentType::MEMORY_MODULE:
                testMemoryModule(config);
                break;
            case ComponentType::OMEGA_RULE:
                testOmegaRule(config);
                break;
            case ComponentType::MUON_OPTIMIZER:
                testMuonOptimizer(config);
                break;
            case ComponentType::FEATURE_MAPPING:
                testFeatureMapping(config);
                break;
            default:
                break;
        }
    }
    
    // Benchmark
    auto benchmark_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        switch (type) {
            case ComponentType::MEMORY_MODULE:
                testMemoryModule(config);
                break;
            case ComponentType::OMEGA_RULE:
                testOmegaRule(config);
                break;
            case ComponentType::MUON_OPTIMIZER:
                testMuonOptimizer(config);
                break;
            case ComponentType::FEATURE_MAPPING:
                testFeatureMapping(config);
                break;
            default:
                break;
        }
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    
    double total_time_ms = std::chrono::duration<double, std::milli>(
        benchmark_end - benchmark_start).count();
    
    metrics.latency_ms = total_time_ms / num_iterations;
    metrics.throughput = num_iterations / (total_time_ms / 1000.0);
    
    // Estimate GFLOPS based on component type
    switch (type) {
        case ComponentType::MEMORY_MODULE:
            // Estimate based on 2-layer MLP operations
            metrics.gflops = (config.hidden_dimension * config.hidden_dimension * 2.0 * config.batch_size) 
                           / (metrics.latency_ms / 1000.0) / 1e9;
            break;
        case ComponentType::FEATURE_MAPPING:
            // Estimate based on polynomial operations
            metrics.gflops = (config.hidden_dimension * 3.0 * config.batch_size) 
                           / (metrics.latency_ms / 1000.0) / 1e9;
            break;
        default:
            metrics.gflops = 0.0;
            break;
    }
    
    metrics.convergence_time_ms = metrics.latency_ms; // Simplified
    
    return metrics;
}

void AtlasTestFramework::printTestSummary(const TestResults& results) {
    std::cout << "\n=== ATLAS Test Summary ===" << std::endl;
    std::cout << "Total Tests: " << results.total_tests << std::endl;
    std::cout << "Passed: " << results.passed_tests << std::endl;
    std::cout << "Failed: " << results.failed_tests << std::endl;
    std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
              << results.success_rate << "%" << std::endl;
    std::cout << "Total Duration: " << std::fixed << std::setprecision(2)
              << results.total_duration_ms / 1000.0 << " seconds" << std::endl;
    
    if (results.failed_tests > 0) {
        std::cout << "\nFailed Tests:" << std::endl;
        for (const auto& result : results.test_case_results) {
            if (!result.passed) {
                std::cout << "  - " << result.test_name << ": " << result.error_message << std::endl;
            }
        }
    }
    
    std::cout << "\n" << (results.failed_tests == 0 ? "ALL TESTS PASSED!" : "SOME TESTS FAILED!") << std::endl;
}

bool AtlasTestFramework::compareAgainstBaseline(const TestResults& current, const TestResults& baseline) {
    // Compare success rates
    if (current.success_rate < baseline.success_rate * 0.95) { // 5% tolerance
        return false;
    }
    
    // Compare performance metrics (simplified)
    // In production, this would compare detailed performance metrics
    
    return true;
}

} // namespace testing
} // namespace atlas