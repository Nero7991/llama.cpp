#!/bin/bash

# ATLAS Testing Framework Runner Script
# This script builds and runs the comprehensive ATLAS test suite

set -e

echo "=== ATLAS Testing Framework Runner ==="
echo "Timestamp: $(date)"
echo ""

# Configuration
BUILD_DIR="build"
VERBOSE=${VERBOSE:-0}
QUICK_TEST=${QUICK_TEST:-0}
PERFORMANCE_TEST=${PERFORMANCE_TEST:-0}
STRESS_TEST=${STRESS_TEST:-0}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --quick)
            QUICK_TEST=1
            shift
            ;;
        --performance)
            PERFORMANCE_TEST=1
            shift
            ;;
        --stress)
            STRESS_TEST=1
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --verbose, -v     Enable verbose output"
            echo "  --quick           Run only quick unit tests"
            echo "  --performance     Run performance benchmarks"
            echo "  --stress          Run stress tests (time consuming)"
            echo "  --build-dir DIR   Specify build directory (default: build)"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "1. Building ATLAS testing framework..."

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed"
    exit 1
fi

# Build the test executables
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build completed successfully"
echo ""

# Track test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_executable="$2"
    local test_args="${3:-}"
    
    echo "Running $test_name..."
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ $VERBOSE -eq 1 ]; then
        echo "Command: $test_executable $test_args"
    fi
    
    if $test_executable $test_args; then
        echo "✅ $test_name PASSED"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo "❌ $test_name FAILED"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    echo ""
}

# Run tests based on configuration
echo "2. Running ATLAS tests..."

if [ $QUICK_TEST -eq 1 ]; then
    echo "=== Quick Test Mode ==="
    run_test "Unit Tests" "./test-atlas-comprehensive" "--unit-only"
    
elif [ $PERFORMANCE_TEST -eq 1 ]; then
    echo "=== Performance Test Mode ==="
    run_test "Performance Tests" "./test-atlas-comprehensive" "--performance-only"
    run_test "Benchmark Suite" "./test-atlas-benchmark"
    
elif [ $STRESS_TEST -eq 1 ]; then
    echo "=== Stress Test Mode ==="
    run_test "Stress Tests" "./test-atlas-comprehensive" "--stress --verbose"
    
else
    echo "=== Comprehensive Test Mode ==="
    
    # Core unit tests
    run_test "Comprehensive Unit Tests" "./test-atlas-comprehensive" "--unit-only"
    
    # Component-specific tests  
    run_test "Memory Module Tests" "./test-memory-module"
    run_test "Omega Rule Tests" "./test-omega-rule"
    
    # Integration tests
    run_test "Integration Tests" "./test-atlas-integration"
    
    # Performance validation
    if [ $VERBOSE -eq 1 ]; then
        run_test "Performance Tests" "./test-atlas-comprehensive" "--performance-only --verbose"
    else
        run_test "Performance Tests" "./test-atlas-comprehensive" "--performance-only"
    fi
fi

# Optional CUDA tests if available
if [ -f "./test-atlas-cuda-performance" ]; then
    echo "=== CUDA Tests Available ==="
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        echo ""
        
        run_test "CUDA Performance Tests" "./test-atlas-cuda-performance"
    else
        echo "⚠️  CUDA tests built but no NVIDIA GPU detected"
        echo ""
    fi
fi

# Generate summary report
echo "=== TEST RESULTS SUMMARY ==="
echo "Total Tests Run: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    echo "Success Rate: 100%"
    echo "🎉 ALL TESTS PASSED!"
    
    # Optional benchmark run
    if [ $QUICK_TEST -eq 0 ] && [ $PERFORMANCE_TEST -eq 0 ]; then
        echo ""
        echo "3. Running quick benchmark for baseline metrics..."
        ./test-atlas-benchmark > ../atlas_benchmark_$(date +%Y%m%d_%H%M%S).log 2>&1 || true
        echo "✅ Benchmark results saved"
    fi
    
    exit_code=0
else
    success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate: ${success_rate}%"
    echo "❌ $FAILED_TESTS TEST(S) FAILED!"
    exit_code=1
fi

echo ""
echo "Test run completed at $(date)"
echo "Build directory: $(pwd)"

# Optional cleanup suggestion
if [ $exit_code -eq 0 ] && [ $QUICK_TEST -eq 0 ]; then
    echo ""
    echo "💡 To run individual tests:"
    echo "   cd $BUILD_DIR"
    echo "   ./test-atlas-comprehensive --help"
    echo "   ./test-memory-module"
    echo "   ./test-omega-rule"  
    echo "   ./test-atlas-integration"
    echo "   ./test-atlas-benchmark"
fi

exit $exit_code