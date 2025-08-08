#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iomanip>

#include "ggml.h"
#include "llama-atlas.h"

namespace atlas {
namespace testing {

// Test types
enum class TestType {
    UNIT = 0,
    INTEGRATION,
    PERFORMANCE,
    STRESS
};

// Component types for testing
enum class ComponentType {
    MEMORY_MODULE = 0,
    OMEGA_RULE,
    MUON_OPTIMIZER,
    NEWTON_SCHULZ,
    FEATURE_MAPPING,
    FULL_PIPELINE,
    LLAMA_INTEGRATION
};

// Test configuration
struct TestConfig {
    int batch_size = 4;
    int sequence_length = 512;
    int hidden_dimension = 1024;
    int memory_depth = 256;
    int window_size = 128;
    int polynomial_degree = 3;
    int newton_schulz_iterations = 5;
    float learning_rate = 0.001f;
    float tolerance = 1e-5f;
};

// Expected test results
struct ExpectedResults {
    double expected_accuracy = 0.95;
    double expected_performance = 100.0; // tokens/sec or other metric
    double max_memory_usage_mb = 1000.0;
    double max_latency_ms = 100.0;
};

// Test tolerance settings
struct ToleranceSettings {
    float numerical_tolerance = 1e-5f;
    float performance_tolerance = 0.1f; // 10% tolerance
    float memory_tolerance = 0.2f;      // 20% tolerance
    double minimum_success_rate = 0.99; // 99% success rate for stress tests
};

// Test case definition
struct AtlasTestCase {
    std::string name;
    std::string description;
    TestType type;
    ComponentType component;
    TestConfig config;
    ExpectedResults expected;
    ToleranceSettings tolerance;
    int priority = 1; // 1=Critical, 2=Important, 3=Nice-to-have
};

// Performance metrics
struct PerformanceMetrics {
    double latency_ms = 0.0;
    double throughput = 0.0;           // operations/sec
    double gflops = 0.0;
    double memory_bandwidth_gbps = 0.0;
    double gpu_utilization_percent = 0.0;
    double memory_usage_mb = 0.0;
    double convergence_time_ms = 0.0;
    double tokens_per_second = 0.0;
    double power_consumption_watts = 0.0;
};

// Performance baselines
struct PerformanceBaseline {
    double memory_module_gflops = 500.0;
    double omega_rule_ms = 10.0;
    double newton_schulz_ms = 5.0;
    double feature_mapping_vectors_per_sec = 1000000.0;
    double full_pipeline_tokens_per_sec = 100.0;
};

// Test case result
struct TestCaseResult {
    std::string test_name;
    bool passed = false;
    std::string error_message;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    double duration_ms = 0.0;
    PerformanceMetrics performance_metrics;
};

// Overall test results
struct TestResults {
    std::vector<TestCaseResult> test_case_results;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    double total_duration_ms = 0.0;
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    double success_rate = 0.0;
};

// Main test framework class
class AtlasTestFramework {
public:
    AtlasTestFramework();
    ~AtlasTestFramework();
    
    // Test environment setup
    void setupAtlasTestEnvironment();
    void teardownAtlasTestEnvironment();
    
    // Test case management
    bool addTestCase(const AtlasTestCase& test_case);
    TestResults runAllTests();
    TestCaseResult runTestCase(const AtlasTestCase& test_case);
    
    // Component testing
    bool testMemoryModule(const TestConfig& config);
    bool testOmegaRule(const TestConfig& config);
    bool testMuonOptimizer(const TestConfig& config);
    bool testFeatureMapping(const TestConfig& config);
    
    // Integration testing
    bool testFullAtlasPipeline(const TestConfig& config);
    bool testLlamaIntegration(const TestConfig& config);
    
    // Performance testing
    PerformanceMetrics benchmarkComponent(ComponentType type, const TestConfig& config);
    
    // Regression testing
    bool compareAgainstBaseline(const TestResults& current, const TestResults& baseline);
    
    // Reporting
    void printTestSummary(const TestResults& results);
    
private:
    // Test execution methods
    bool runUnitTest(const AtlasTestCase& test_case);
    bool runIntegrationTest(const AtlasTestCase& test_case);
    TestCaseResult runPerformanceTest(const AtlasTestCase& test_case);
    bool runStressTest(const AtlasTestCase& test_case);
    
    // Helper methods
    void loadPerformanceBaselines();
    
    // Test context and resources
    struct atlas_context* test_context_;
    struct ggml_context* ggml_ctx_;
    
    // Test case storage
    std::vector<AtlasTestCase> test_cases_;
    
    // Performance baselines
    PerformanceBaseline baseline_metrics_;
    
    // Test statistics
    int total_tests_;
    int passed_tests_;
    int failed_tests_;
};

// Utility functions
inline std::string testTypeToString(TestType type) {
    switch (type) {
        case TestType::UNIT: return "Unit";
        case TestType::INTEGRATION: return "Integration";
        case TestType::PERFORMANCE: return "Performance";
        case TestType::STRESS: return "Stress";
        default: return "Unknown";
    }
}

inline std::string componentTypeToString(ComponentType type) {
    switch (type) {
        case ComponentType::MEMORY_MODULE: return "Memory Module";
        case ComponentType::OMEGA_RULE: return "Omega Rule";
        case ComponentType::MUON_OPTIMIZER: return "Muon Optimizer";
        case ComponentType::NEWTON_SCHULZ: return "Newton-Schulz";
        case ComponentType::FEATURE_MAPPING: return "Feature Mapping";
        case ComponentType::FULL_PIPELINE: return "Full Pipeline";
        case ComponentType::LLAMA_INTEGRATION: return "Llama Integration";
        default: return "Unknown";
    }
}

// Test builder utility class
class AtlasTestBuilder {
public:
    AtlasTestBuilder& name(const std::string& test_name) {
        test_case_.name = test_name;
        return *this;
    }
    
    AtlasTestBuilder& description(const std::string& desc) {
        test_case_.description = desc;
        return *this;
    }
    
    AtlasTestBuilder& type(TestType test_type) {
        test_case_.type = test_type;
        return *this;
    }
    
    AtlasTestBuilder& component(ComponentType comp_type) {
        test_case_.component = comp_type;
        return *this;
    }
    
    AtlasTestBuilder& config(const TestConfig& cfg) {
        test_case_.config = cfg;
        return *this;
    }
    
    AtlasTestBuilder& expected(const ExpectedResults& exp) {
        test_case_.expected = exp;
        return *this;
    }
    
    AtlasTestBuilder& tolerance(const ToleranceSettings& tol) {
        test_case_.tolerance = tol;
        return *this;
    }
    
    AtlasTestBuilder& priority(int prio) {
        test_case_.priority = prio;
        return *this;
    }
    
    AtlasTestCase build() {
        return test_case_;
    }
    
private:
    AtlasTestCase test_case_;
};

} // namespace testing
} // namespace atlas