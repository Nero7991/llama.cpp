#include "framework/atlas-test-framework.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace atlas::testing;

// Test configuration presets
TestConfig createSmallConfig() {
    TestConfig config;
    config.batch_size = 2;
    config.sequence_length = 128;
    config.hidden_dimension = 512;
    config.memory_depth = 128;
    config.window_size = 64;
    return config;
}

TestConfig createMediumConfig() {
    TestConfig config;
    config.batch_size = 4;
    config.sequence_length = 1024;
    config.hidden_dimension = 2048;
    config.memory_depth = 512;
    config.window_size = 128;
    return config;
}

TestConfig createLargeConfig() {
    TestConfig config;
    config.batch_size = 8;
    config.sequence_length = 4096;
    config.hidden_dimension = 4096;
    config.memory_depth = 1024;
    config.window_size = 256;
    return config;
}

// Create comprehensive test suite
std::vector<AtlasTestCase> createTestSuite() {
    std::vector<AtlasTestCase> test_cases;
    
    // Unit Tests - Critical Priority
    
    // Memory Module Unit Tests
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MemoryModule_SmallInput_UnitTest")
            .description("Test memory module with small input dimensions")
            .type(TestType::UNIT)
            .component(ComponentType::MEMORY_MODULE)
            .config(createSmallConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MemoryModule_MediumInput_UnitTest")
            .description("Test memory module with medium input dimensions")
            .type(TestType::UNIT)
            .component(ComponentType::MEMORY_MODULE)
            .config(createMediumConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MemoryModule_LargeInput_UnitTest")
            .description("Test memory module with large input dimensions")
            .type(TestType::UNIT)
            .component(ComponentType::MEMORY_MODULE)
            .config(createLargeConfig())
            .priority(2)
            .build()
    );
    
    // Omega Rule Unit Tests
    test_cases.push_back(
        AtlasTestBuilder()
            .name("OmegaRule_Basic_UnitTest")
            .description("Test basic Omega Rule functionality")
            .type(TestType::UNIT)
            .component(ComponentType::OMEGA_RULE)
            .config(createSmallConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("OmegaRule_LargeWindow_UnitTest")
            .description("Test Omega Rule with large window size")
            .type(TestType::UNIT)
            .component(ComponentType::OMEGA_RULE)
            .config(createMediumConfig())
            .priority(2)
            .build()
    );
    
    // Muon Optimizer Unit Tests
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MuonOptimizer_Basic_UnitTest")
            .description("Test basic Muon optimizer functionality")
            .type(TestType::UNIT)
            .component(ComponentType::MUON_OPTIMIZER)
            .config(createSmallConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MuonOptimizer_Convergence_UnitTest")
            .description("Test Muon optimizer convergence properties")
            .type(TestType::UNIT)
            .component(ComponentType::MUON_OPTIMIZER)
            .config(createMediumConfig())
            .priority(1)
            .build()
    );
    
    // Feature Mapping Unit Tests
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FeatureMapping_Polynomial_UnitTest")
            .description("Test polynomial feature mapping")
            .type(TestType::UNIT)
            .component(ComponentType::FEATURE_MAPPING)
            .config(createSmallConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FeatureMapping_HighDegree_UnitTest")
            .description("Test high-degree polynomial feature mapping")
            .type(TestType::UNIT)
            .component(ComponentType::FEATURE_MAPPING)
            .config(createMediumConfig())
            .priority(2)
            .build()
    );
    
    // Integration Tests - Important Priority
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FullPipeline_Small_IntegrationTest")
            .description("Test complete ATLAS pipeline with small model")
            .type(TestType::INTEGRATION)
            .component(ComponentType::FULL_PIPELINE)
            .config(createSmallConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FullPipeline_Medium_IntegrationTest")
            .description("Test complete ATLAS pipeline with medium model")
            .type(TestType::INTEGRATION)
            .component(ComponentType::FULL_PIPELINE)
            .config(createMediumConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FullPipeline_Large_IntegrationTest")
            .description("Test complete ATLAS pipeline with large model")
            .type(TestType::INTEGRATION)
            .component(ComponentType::FULL_PIPELINE)
            .config(createLargeConfig())
            .priority(2)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("LlamaIntegration_Basic_IntegrationTest")
            .description("Test ATLAS integration with llama.cpp")
            .type(TestType::INTEGRATION)
            .component(ComponentType::LLAMA_INTEGRATION)
            .config(createMediumConfig())
            .priority(1)
            .build()
    );
    
    // Performance Tests - Important Priority
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MemoryModule_Performance_Benchmark")
            .description("Benchmark memory module performance")
            .type(TestType::PERFORMANCE)
            .component(ComponentType::MEMORY_MODULE)
            .config(createLargeConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("OmegaRule_Performance_Benchmark")
            .description("Benchmark Omega Rule performance")
            .type(TestType::PERFORMANCE)
            .component(ComponentType::OMEGA_RULE)
            .config(createLargeConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MuonOptimizer_Performance_Benchmark")
            .description("Benchmark Muon optimizer performance")
            .type(TestType::PERFORMANCE)
            .component(ComponentType::MUON_OPTIMIZER)
            .config(createLargeConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FeatureMapping_Performance_Benchmark")
            .description("Benchmark feature mapping performance")
            .type(TestType::PERFORMANCE)
            .component(ComponentType::FEATURE_MAPPING)
            .config(createLargeConfig())
            .priority(1)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FullPipeline_Performance_Benchmark")
            .description("Benchmark complete ATLAS pipeline performance")
            .type(TestType::PERFORMANCE)
            .component(ComponentType::FULL_PIPELINE)
            .config(createLargeConfig())
            .priority(1)
            .build()
    );
    
    // Stress Tests - Lower Priority
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("MemoryModule_Stress_Test")
            .description("Stress test memory module reliability")
            .type(TestType::STRESS)
            .component(ComponentType::MEMORY_MODULE)
            .config(createMediumConfig())
            .priority(2)
            .build()
    );
    
    test_cases.push_back(
        AtlasTestBuilder()
            .name("FullPipeline_Stress_Test")
            .description("Stress test full pipeline reliability")
            .type(TestType::STRESS)
            .component(ComponentType::FULL_PIPELINE)
            .config(createLargeConfig())
            .priority(2)
            .build()
    );
    
    return test_cases;
}

void printTestConfiguration() {
    std::cout << "=== ATLAS Comprehensive Test Suite ===" << std::endl;
    std::cout << "Test Configurations:" << std::endl;
    std::cout << "  Small:  batch=2,  seq=128,  hidden=512,  memory=128" << std::endl;
    std::cout << "  Medium: batch=4,  seq=1024, hidden=2048, memory=512" << std::endl;
    std::cout << "  Large:  batch=8,  seq=4096, hidden=4096, memory=1024" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    bool run_unit_tests = true;
    bool run_integration_tests = true;
    bool run_performance_tests = true;
    bool run_stress_tests = false; // Disabled by default due to time
    bool verbose = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--unit-only") {
            run_integration_tests = false;
            run_performance_tests = false;
            run_stress_tests = false;
        } else if (arg == "--performance-only") {
            run_unit_tests = false;
            run_integration_tests = false;
            run_stress_tests = false;
        } else if (arg == "--stress") {
            run_stress_tests = true;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --unit-only      Run only unit tests" << std::endl;
            std::cout << "  --performance-only  Run only performance tests" << std::endl;
            std::cout << "  --stress         Include stress tests" << std::endl;
            std::cout << "  --verbose, -v    Verbose output" << std::endl;
            std::cout << "  --help, -h       Show this help" << std::endl;
            return 0;
        }
    }
    
    try {
        printTestConfiguration();
        
        // Initialize test framework
        AtlasTestFramework framework;
        
        // Create and add test cases
        auto test_cases = createTestSuite();
        
        int added_tests = 0;
        for (const auto& test_case : test_cases) {
            // Filter test cases based on command line options
            bool should_run = false;
            
            switch (test_case.type) {
                case TestType::UNIT:
                    should_run = run_unit_tests;
                    break;
                case TestType::INTEGRATION:
                    should_run = run_integration_tests;
                    break;
                case TestType::PERFORMANCE:
                    should_run = run_performance_tests;
                    break;
                case TestType::STRESS:
                    should_run = run_stress_tests;
                    break;
            }
            
            if (should_run) {
                framework.addTestCase(test_case);
                added_tests++;
                
                if (verbose) {
                    std::cout << "Added test: " << test_case.name 
                              << " (" << testTypeToString(test_case.type) 
                              << ", " << componentTypeToString(test_case.component) << ")" << std::endl;
                }
            }
        }
        
        std::cout << "Running " << added_tests << " test cases..." << std::endl;
        
        // Run all tests
        TestResults results = framework.runAllTests();
        
        // Determine exit code
        int exit_code = (results.failed_tests == 0) ? 0 : 1;
        
        // Print final status
        std::cout << "\n" << std::string(50, '=') << std::endl;
        if (results.failed_tests == 0) {
            std::cout << "🎉 ALL ATLAS TESTS PASSED!" << std::endl;
        } else {
            std::cout << "❌ " << results.failed_tests << " TEST(S) FAILED!" << std::endl;
        }
        std::cout << std::string(50, '=') << std::endl;
        
        return exit_code;
        
    } catch (const std::exception& e) {
        std::cerr << "Test framework error: " << e.what() << std::endl;
        return 1;
    }
}

// Additional helper functions for specific test scenarios

namespace atlas {
namespace testing {

// Create edge case test configurations
TestConfig createEdgeCaseConfig() {
    TestConfig config;
    config.batch_size = 1;
    config.sequence_length = 1;
    config.hidden_dimension = 32;
    config.memory_depth = 16;
    config.window_size = 8;
    return config;
}

// Create memory pressure test configuration
TestConfig createMemoryPressureConfig() {
    TestConfig config;
    config.batch_size = 16;
    config.sequence_length = 8192;
    config.hidden_dimension = 8192;
    config.memory_depth = 2048;
    config.window_size = 512;
    return config;
}

// Validate numerical stability
bool validateNumericalStability(const std::vector<float>& values, float = 1e-5f) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false; // NaN or infinity detected
        }
        
        if (std::abs(value) > 1e6f) {
            return false; // Unreasonably large values
        }
    }
    return true;
}

// Calculate relative error
double calculateRelativeError(double computed, double expected) {
    if (std::abs(expected) < 1e-10) {
        return std::abs(computed - expected);
    }
    return std::abs(computed - expected) / std::abs(expected);
}

// Performance regression detector
bool detectPerformanceRegression(const PerformanceMetrics& current, 
                                const PerformanceMetrics& baseline, 
                                double threshold = 0.05) {
    // Check if performance degraded by more than threshold
    if (current.throughput < baseline.throughput * (1.0 - threshold)) {
        return true;
    }
    
    if (current.latency_ms > baseline.latency_ms * (1.0 + threshold)) {
        return true;
    }
    
    if (current.memory_usage_mb > baseline.memory_usage_mb * (1.0 + threshold)) {
        return true;
    }
    
    return false;
}

} // namespace testing
} // namespace atlas