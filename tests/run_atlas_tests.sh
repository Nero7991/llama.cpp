#!/bin/bash

# ATLAS Persistence Test Runner
# Comprehensive test suite for Issue #12 - ATLAS Phase 6B Memory Persistence

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
LLAMA_CPP_DIR="/home/orencollaco/GitHub/llama.cpp"
TESTS_DIR="${LLAMA_CPP_DIR}/tests"
BUILD_DIR="${TESTS_DIR}/build"
INCLUDE_DIR="${LLAMA_CPP_DIR}/include"
SRC_DIR="${LLAMA_CPP_DIR}/src"

echo -e "${YELLOW}=== ATLAS Persistence Test Suite ===${NC}"
echo "Build directory: ${BUILD_DIR}"
echo "Source directory: ${LLAMA_CPP_DIR}"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"

# Function to compile and run a test
run_test() {
    local test_name="$1"
    local test_file="$2"
    
    echo -e "${YELLOW}Building ${test_name}...${NC}"
    
    # Compile the test
    cd "${TESTS_DIR}"
    g++ -std=c++11 -I"${INCLUDE_DIR}" -o "${BUILD_DIR}/${test_name}" \
        "${test_file}" "${SRC_DIR}/atlas-persistence.cpp" \
        -lpthread || {
        echo -e "${RED}❌ Failed to compile ${test_name}${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ Compiled ${test_name}${NC}"
    
    # Run the test
    echo -e "${YELLOW}Running ${test_name}...${NC}"
    cd "${BUILD_DIR}"
    "./${test_name}" || {
        echo -e "${RED}❌ ${test_name} FAILED${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ ${test_name} PASSED${NC}"
    echo ""
}

# Function to run basic functionality test
test_basic_functionality() {
    echo -e "${YELLOW}=== Basic Functionality Test ===${NC}"
    
    cd "${LLAMA_CPP_DIR}/examples/server"
    if [ -f "simple-server" ]; then
        echo "Testing simple server demo..."
        ./simple-server -m test-model.gguf || {
            echo -e "${RED}❌ Simple server test failed${NC}"
            return 1
        }
        echo -e "${GREEN}✓ Simple server test passed${NC}"
    else
        echo "Building simple server..."
        g++ -std=c++11 -I../../include -o simple-server simple-server.cpp ../../src/atlas-persistence.cpp || {
            echo -e "${RED}❌ Failed to build simple server${NC}"
            return 1
        }
        echo "Testing simple server demo..."
        ./simple-server -m test-model.gguf || {
            echo -e "${RED}❌ Simple server test failed${NC}"
            return 1
        }
        echo -e "${GREEN}✓ Simple server test passed${NC}"
    fi
    echo ""
}

# Function to test file format validation
test_file_format() {
    echo -e "${YELLOW}=== File Format Validation Test ===${NC}"
    
    # Create a test file and verify its format
    cd "${BUILD_DIR}"
    
    # Create test data
    cat > test_format_validation.cpp << 'EOF'
#include "../../../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>

int main() {
    const char* test_file = "/tmp/format_test.atlas";
    
    // Create and save a file
    atlas_persistence_t* ctx = atlas_persistence_create();
    atlas_persistence_set(ctx, "format_test", "ATLAS_FORMAT_TEST", 17, 1);
    atlas_persistence_save(ctx, test_file);
    atlas_persistence_free(ctx);
    
    // Read and verify binary format
    FILE* file = fopen(test_file, "rb");
    if (!file) {
        printf("Failed to open test file\n");
        return 1;
    }
    
    // Check magic number
    uint32_t magic;
    if (fread(&magic, sizeof(magic), 1, file) != 1) {
        printf("Failed to read magic number\n");
        return 1;
    }
    
    fclose(file);
    unlink(test_file);
    
    if (magic == 0x534C5441) {  // "ATLS" in little endian
        printf("✓ File format validation: PASSED\n");
        return 0;
    } else {
        printf("❌ File format validation: FAILED (magic: 0x%08X)\n", magic);
        return 1;
    }
}
EOF
    
    g++ -std=c++11 -I"${INCLUDE_DIR}" -o test_format_validation \
        test_format_validation.cpp "${SRC_DIR}/atlas-persistence.cpp" || {
        echo -e "${RED}❌ Failed to compile format validation test${NC}"
        return 1
    }
    
    ./test_format_validation || {
        echo -e "${RED}❌ File format validation failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ File format validation passed${NC}"
    echo ""
}

# Function to test memory stress scenarios
test_memory_stress() {
    echo -e "${YELLOW}=== Memory Stress Test ===${NC}"
    
    cd "${BUILD_DIR}"
    
    cat > memory_stress_test.cpp << 'EOF'
#include "../../../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    printf("Testing memory stress scenarios...\n");
    
    atlas_persistence_t* ctx = atlas_persistence_create();
    if (!ctx) {
        printf("Failed to create context\n");
        return 1;
    }
    
    // Test 1: Many small entries
    printf("Adding 5000 small entries... ");
    for (int i = 0; i < 5000; i++) {
        char key[32], value[64];
        snprintf(key, sizeof(key), "stress_key_%d", i);
        snprintf(value, sizeof(value), "stress_value_%d_data", i);
        
        if (!atlas_persistence_set(ctx, key, value, strlen(value), i % 5)) {
            printf("Failed at entry %d\n", i);
            return 1;
        }
    }
    printf("OK\n");
    
    // Test 2: Large entry
    printf("Adding large entry (1MB)... ");
    char* large_data = (char*)malloc(1024 * 1024);
    if (!large_data) {
        printf("Failed to allocate large data\n");
        return 1;
    }
    memset(large_data, 'X', 1024 * 1024);
    
    if (!atlas_persistence_set(ctx, "large_entry", large_data, 1024 * 1024, 99)) {
        printf("Failed to store large entry\n");
        free(large_data);
        return 1;
    }
    free(large_data);
    printf("OK\n");
    
    printf("Total entries: %zu\n", atlas_persistence_count(ctx));
    
    // Test 3: Save/load large dataset
    printf("Saving large dataset... ");
    const char* test_file = "/tmp/stress_test.atlas";
    if (!atlas_persistence_save(ctx, test_file)) {
        printf("Failed to save\n");
        return 1;
    }
    printf("OK\n");
    
    atlas_persistence_free(ctx);
    
    // Test 4: Load and verify
    printf("Loading and verifying... ");
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    if (!atlas_persistence_load(ctx2, test_file)) {
        printf("Failed to load\n");
        return 1;
    }
    
    if (atlas_persistence_count(ctx2) != 5001) {  // 5000 small + 1 large
        printf("Count mismatch: expected 5001, got %zu\n", atlas_persistence_count(ctx2));
        return 1;
    }
    
    atlas_persistence_free(ctx2);
    unlink(test_file);
    printf("OK\n");
    
    printf("✓ Memory stress test: PASSED\n");
    return 0;
}
EOF
    
    g++ -std=c++11 -I"${INCLUDE_DIR}" -o memory_stress_test \
        memory_stress_test.cpp "${SRC_DIR}/atlas-persistence.cpp" || {
        echo -e "${RED}❌ Failed to compile memory stress test${NC}"
        return 1
    }
    
    ./memory_stress_test || {
        echo -e "${RED}❌ Memory stress test failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}✓ Memory stress test passed${NC}"
    echo ""
}

# Main test execution
main() {
    local tests_passed=0
    local tests_total=0
    
    # Check if source files exist
    if [ ! -f "${INCLUDE_DIR}/atlas-persistence.h" ]; then
        echo -e "${RED}❌ atlas-persistence.h not found at ${INCLUDE_DIR}${NC}"
        exit 1
    fi
    
    if [ ! -f "${SRC_DIR}/atlas-persistence.cpp" ]; then
        echo -e "${RED}❌ atlas-persistence.cpp not found at ${SRC_DIR}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Source files found${NC}"
    echo ""
    
    # Run basic functionality test
    if test_basic_functionality; then
        ((tests_passed++))
    fi
    ((tests_total++))
    
    # Run file format validation
    if test_file_format; then
        ((tests_passed++))
    fi
    ((tests_total++))
    
    # Run unit tests
    if run_test "unit-tests" "test-atlas-persistence-unit.cpp"; then
        ((tests_passed++))
    fi
    ((tests_total++))
    
    # Run integration tests
    if run_test "integration-tests" "test-atlas-persistence-integration.cpp"; then
        ((tests_passed++))
    fi
    ((tests_total++))
    
    # Run memory stress test
    if test_memory_stress; then
        ((tests_passed++))
    fi
    ((tests_total++))
    
    # Summary
    echo -e "${YELLOW}=== Test Summary ===${NC}"
    echo "Tests passed: ${tests_passed}/${tests_total}"
    
    if [ $tests_passed -eq $tests_total ]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! ATLAS Persistence is ready for production.${NC}"
        return 0
    else
        echo -e "${RED}❌ Some tests failed. Please review the output above.${NC}"
        return 1
    fi
}

# Clean up function
cleanup() {
    echo -e "${YELLOW}Cleaning up test files...${NC}"
    rm -f /tmp/test_atlas.atlas
    rm -f /tmp/test-model_memory.atlas
    rm -f /tmp/format_test.atlas
    rm -f /tmp/stress_test.atlas
    rm -f /tmp/large_dataset.atlas
    rm -f /tmp/concurrent_test.atlas
    rm -f /tmp/corruption_test.atlas
    rm -f /tmp/cross_platform.atlas
    rm -f /tmp/performance_test.atlas
}

# Set up signal handling for cleanup
trap cleanup EXIT

# Run main test suite
main "$@"