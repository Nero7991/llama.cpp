#!/bin/bash

# ATLAS Test Runner Script
# Runs all ATLAS tests with appropriate configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build"
TEST_DIR="${BUILD_DIR}/bin"

# Default settings
VERBOSE=false
CPU_ONLY=false
QUICK_MODE=false
PARALLEL_JOBS=4

# Test categories
ATLAS_TESTS=(
    "test-atlas-types"
    "test-atlas-backend" 
    "test-atlas-memory"
    "test-atlas-operations"
    "test-atlas-integration"
    "test-atlas-configurations"
)

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_status $BLUE "=================================="
    print_status $BLUE "$1"
    print_status $BLUE "=================================="
    echo
}

print_test_result() {
    local test_name=$1
    local result=$2
    local time_taken=$3
    
    if [ "$result" -eq 0 ]; then
        print_status $GREEN "[PASS] $test_name (${time_taken}s)"
    else
        print_status $RED "[FAIL] $test_name (${time_taken}s)"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [options]

ATLAS Test Runner - Comprehensive testing for ATLAS Phase 1 implementation

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -c, --cpu-only      Run CPU-only tests (skip CUDA/GPU tests)
    -q, --quick         Quick mode - run essential tests only
    -j, --jobs N        Number of parallel jobs for building (default: 4)
    -b, --build-dir     Specify build directory (default: ../build)
    -t, --test NAME     Run specific test only
    --list-tests        List available ATLAS tests
    --check-deps        Check test dependencies
    --benchmark         Run performance benchmarks

Examples:
    $0                              # Run all ATLAS tests
    $0 --cpu-only                   # Run CPU tests only
    $0 --test test-atlas-memory     # Run memory tests only
    $0 --quick --verbose            # Quick test with verbose output

EOF
}

# Function to list available tests
list_tests() {
    print_header "Available ATLAS Tests"
    for test in "${ATLAS_TESTS[@]}"; do
        echo "  - $test"
    done
    echo
}

# Function to check dependencies
check_dependencies() {
    print_header "Checking ATLAS Test Dependencies"
    
    local deps_ok=true
    
    # Check if build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        print_status $RED "❌ Build directory not found: $BUILD_DIR"
        deps_ok=false
    else
        print_status $GREEN "✅ Build directory found"
    fi
    
    # Check if CMake was run
    if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
        print_status $RED "❌ CMake not configured. Run: cmake -B build"
        deps_ok=false
    else
        print_status $GREEN "✅ CMake configured"
    fi
    
    # Check for test executables
    local missing_tests=()
    for test in "${ATLAS_TESTS[@]}"; do
        if [ ! -f "$TEST_DIR/$test" ]; then
            missing_tests+=("$test")
        fi
    done
    
    if [ ${#missing_tests[@]} -eq 0 ]; then
        print_status $GREEN "✅ All ATLAS test executables found"
    else
        print_status $RED "❌ Missing test executables: ${missing_tests[*]}"
        print_status $YELLOW "   Run: cmake --build build --target <test_name>"
        deps_ok=false
    fi
    
    # Check system capabilities
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_status $GREEN "✅ NVIDIA GPU detected"
        else
            print_status $YELLOW "⚠️  NVIDIA driver issues detected"
        fi
    else
        print_status $YELLOW "⚠️  NVIDIA GPU not detected (CUDA tests will be skipped)"
    fi
    
    # Check memory availability
    local available_mem_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_mem_gb" -lt 4 ]; then
        print_status $YELLOW "⚠️  Low available memory (${available_mem_gb}GB). Some tests may fail."
    else
        print_status $GREEN "✅ Sufficient memory available (${available_mem_gb}GB)"
    fi
    
    if [ "$deps_ok" = false ]; then
        print_status $RED "❌ Dependency checks failed"
        exit 1
    else
        print_status $GREEN "✅ All dependency checks passed"
    fi
    
    echo
}

# Function to build tests
build_tests() {
    print_header "Building ATLAS Tests"
    
    if [ ! -d "$BUILD_DIR" ]; then
        print_status $RED "Build directory not found. Creating..."
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        cmake .. || { print_status $RED "CMake configuration failed"; exit 1; }
    fi
    
    cd "$BUILD_DIR"
    
    # Build specific ATLAS tests
    for test in "${ATLAS_TESTS[@]}"; do
        print_status $YELLOW "Building $test..."
        if [ "$VERBOSE" = true ]; then
            cmake --build . --target "$test" -j "$PARALLEL_JOBS"
        else
            cmake --build . --target "$test" -j "$PARALLEL_JOBS" > /dev/null 2>&1
        fi
        
        if [ $? -eq 0 ]; then
            print_status $GREEN "✅ $test built successfully"
        else
            print_status $RED "❌ Failed to build $test"
            exit 1
        fi
    done
    
    echo
}

# Function to run a single test
run_single_test() {
    local test_name=$1
    local test_path="$TEST_DIR/$test_name"
    
    if [ ! -f "$test_path" ]; then
        print_status $RED "Test executable not found: $test_path"
        return 1
    fi
    
    local start_time=$(date +%s)
    
    if [ "$VERBOSE" = true ]; then
        print_status $YELLOW "Running $test_name with verbose output:"
        "$test_path"
        local result=$?
    else
        "$test_path" > /dev/null 2>&1
        local result=$?
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_test_result "$test_name" $result $duration
    
    return $result
}

# Function to run performance benchmarks
run_benchmarks() {
    print_header "ATLAS Performance Benchmarks"
    
    local benchmark_tests=(
        "test-atlas-memory"
        "test-atlas-operations"
        "test-atlas-integration"
    )
    
    for test in "${benchmark_tests[@]}"; do
        print_status $YELLOW "Benchmarking $test..."
        
        # Run test 5 times and calculate average time
        local total_time=0
        local successful_runs=0
        
        for i in {1..5}; do
            local start_time=$(date +%s.%3N)
            "$TEST_DIR/$test" > /dev/null 2>&1
            local result=$?
            local end_time=$(date +%s.%3N)
            
            if [ $result -eq 0 ]; then
                local run_time=$(echo "$end_time - $start_time" | bc)
                total_time=$(echo "$total_time + $run_time" | bc)
                successful_runs=$((successful_runs + 1))
            fi
        done
        
        if [ $successful_runs -gt 0 ]; then
            local avg_time=$(echo "scale=3; $total_time / $successful_runs" | bc)
            print_status $GREEN "Average time: ${avg_time}s ($successful_runs/5 successful runs)"
        else
            print_status $RED "All benchmark runs failed"
        fi
    done
    
    echo
}

# Function to run all tests
run_all_tests() {
    print_header "Running ATLAS Test Suite"
    
    local passed_tests=0
    local total_tests=0
    local failed_tests=()
    
    local tests_to_run=("${ATLAS_TESTS[@]}")
    
    # Filter tests for quick mode
    if [ "$QUICK_MODE" = true ]; then
        tests_to_run=(
            "test-atlas-types"
            "test-atlas-memory"
            "test-atlas-integration"
        )
        print_status $YELLOW "Running in quick mode - selected tests only"
        echo
    fi
    
    for test in "${tests_to_run[@]}"; do
        total_tests=$((total_tests + 1))
        
        if run_single_test "$test"; then
            passed_tests=$((passed_tests + 1))
        else
            failed_tests+=("$test")
        fi
    done
    
    # Print summary
    echo
    print_header "Test Results Summary"
    print_status $GREEN "Passed: $passed_tests/$total_tests tests"
    
    if [ ${#failed_tests[@]} -gt 0 ]; then
        print_status $RED "Failed tests:"
        for test in "${failed_tests[@]}"; do
            echo "  - $test"
        done
        echo
        return 1
    else
        print_status $GREEN "🎉 All ATLAS tests passed!"
        echo
        return 0
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--cpu-only)
            CPU_ONLY=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR="$2"
            TEST_DIR="${BUILD_DIR}/bin"
            shift 2
            ;;
        -t|--test)
            SINGLE_TEST="$2"
            shift 2
            ;;
        --list-tests)
            list_tests
            exit 0
            ;;
        --check-deps)
            check_dependencies
            exit 0
            ;;
        --benchmark)
            RUN_BENCHMARKS=true
            shift
            ;;
        *)
            print_status $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header "ATLAS Test Runner"

# Check dependencies first
check_dependencies

# Build tests
build_tests

# Run specific test if requested
if [ -n "$SINGLE_TEST" ]; then
    print_header "Running Single Test: $SINGLE_TEST"
    if run_single_test "$SINGLE_TEST"; then
        print_status $GREEN "✅ Test passed"
        exit 0
    else
        print_status $RED "❌ Test failed"
        exit 1
    fi
fi

# Run benchmarks if requested
if [ "$RUN_BENCHMARKS" = true ]; then
    if command -v bc &> /dev/null; then
        run_benchmarks
    else
        print_status $RED "bc command not found. Install bc for benchmarks."
        exit 1
    fi
fi

# Run all tests
if run_all_tests; then
    exit 0
else
    exit 1
fi