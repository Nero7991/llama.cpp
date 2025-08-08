# ATLAS Testing Framework CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(AtlasTestingFramework)

# Find GGML and CUDA if available
find_package(CUDA QUIET)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include)

# Source files for the test framework
set(ATLAS_TEST_FRAMEWORK_SOURCES
    framework/atlas-test-framework.cpp
)

# Common libraries to link against
set(ATLAS_TEST_LIBRARIES 
    ggml
    pthread
)

# Add ATLAS implementation sources
set(ATLAS_IMPLEMENTATION_SOURCES
    ../../src/llama-atlas.cpp
)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Add math library
if(UNIX)
    set(ATLAS_TEST_LIBRARIES ${ATLAS_TEST_LIBRARIES} m)
endif()

# Create test framework library
add_library(atlas_test_framework STATIC 
    ${ATLAS_TEST_FRAMEWORK_SOURCES}
)

target_link_libraries(atlas_test_framework ${ATLAS_TEST_LIBRARIES})

# Comprehensive test suite
add_executable(test-atlas-comprehensive
    test-atlas-comprehensive.cpp
    ${ATLAS_IMPLEMENTATION_SOURCES}
)

target_link_libraries(test-atlas-comprehensive 
    atlas_test_framework 
    ${ATLAS_TEST_LIBRARIES}
)

# Memory Module specific tests
add_executable(test-memory-module
    test-memory-module.cpp
    ${ATLAS_IMPLEMENTATION_SOURCES}
)

target_link_libraries(test-memory-module 
    atlas_test_framework 
    ${ATLAS_TEST_LIBRARIES}
)

# Omega Rule specific tests
add_executable(test-omega-rule
    test-omega-rule.cpp
    ${ATLAS_IMPLEMENTATION_SOURCES}
)

target_link_libraries(test-omega-rule 
    atlas_test_framework 
    ${ATLAS_TEST_LIBRARIES}
)

# CUDA-specific tests if CUDA is available
if(CUDA_FOUND)
    enable_language(CUDA)
    
    set(CUDA_TEST_SOURCES
        ../../ggml/src/ggml-cuda/atlas-cuda.cu
        ../../ggml/src/ggml-cuda/atlas-multi-gpu.cu
    )
    
    add_executable(test-atlas-cuda-performance
        test-atlas-cuda-performance.cpp
        ${ATLAS_IMPLEMENTATION_SOURCES}
        ${CUDA_TEST_SOURCES}
    )
    
    target_link_libraries(test-atlas-cuda-performance 
        atlas_test_framework 
        ${ATLAS_TEST_LIBRARIES}
        ${CUDA_LIBRARIES}
    )
    
    set_property(TARGET test-atlas-cuda-performance PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Set CUDA architecture
    set_property(TARGET test-atlas-cuda-performance PROPERTY CUDA_ARCHITECTURES "60;70;75;80;86")
    
endif()

# Integration tests
add_executable(test-atlas-integration
    test-atlas-integration.cpp
    ${ATLAS_IMPLEMENTATION_SOURCES}
)

target_link_libraries(test-atlas-integration 
    atlas_test_framework 
    ${ATLAS_TEST_LIBRARIES}
)

# Performance benchmark
add_executable(test-atlas-benchmark
    test-atlas-benchmark.cpp
    ${ATLAS_IMPLEMENTATION_SOURCES}
)

target_link_libraries(test-atlas-benchmark 
    atlas_test_framework 
    ${ATLAS_TEST_LIBRARIES}
)

# Test runner script
add_custom_target(run-atlas-tests
    COMMAND echo "Running ATLAS comprehensive tests..."
    COMMAND $<TARGET_FILE:test-atlas-comprehensive> --unit-only
    COMMAND echo "Running Memory Module tests..."
    COMMAND $<TARGET_FILE:test-memory-module>
    COMMAND echo "Running Omega Rule tests..."
    COMMAND $<TARGET_FILE:test-omega-rule>
    COMMAND echo "Running Integration tests..."
    COMMAND $<TARGET_FILE:test-atlas-integration>
    DEPENDS test-atlas-comprehensive test-memory-module test-omega-rule test-atlas-integration
)

# Quick test target for CI/CD
add_custom_target(run-atlas-quick-tests
    COMMAND echo "Running ATLAS quick test suite..."
    COMMAND $<TARGET_FILE:test-atlas-comprehensive> --unit-only --verbose
    DEPENDS test-atlas-comprehensive
)

# Performance test target
add_custom_target(run-atlas-performance-tests
    COMMAND echo "Running ATLAS performance tests..."
    COMMAND $<TARGET_FILE:test-atlas-comprehensive> --performance-only
    COMMAND $<TARGET_FILE:test-atlas-benchmark>
    DEPENDS test-atlas-comprehensive test-atlas-benchmark
)

# Stress test target
add_custom_target(run-atlas-stress-tests
    COMMAND echo "Running ATLAS stress tests..."
    COMMAND $<TARGET_FILE:test-atlas-comprehensive> --stress --verbose
    DEPENDS test-atlas-comprehensive
)

# Install test executables (optional)
install(TARGETS 
    test-atlas-comprehensive
    test-memory-module
    test-omega-rule
    test-atlas-integration
    test-atlas-benchmark
    DESTINATION bin/atlas-tests
    OPTIONAL
)

# Testing with CTest integration
enable_testing()

add_test(NAME AtlasUnitTests 
         COMMAND test-atlas-comprehensive --unit-only)

add_test(NAME AtlasMemoryModule 
         COMMAND test-memory-module)

add_test(NAME AtlasOmegaRule 
         COMMAND test-omega-rule)

add_test(NAME AtlasIntegration 
         COMMAND test-atlas-integration)

# Set test properties
set_tests_properties(AtlasUnitTests PROPERTIES TIMEOUT 300)  # 5 minutes
set_tests_properties(AtlasMemoryModule PROPERTIES TIMEOUT 120)  # 2 minutes
set_tests_properties(AtlasOmegaRule PROPERTIES TIMEOUT 120)     # 2 minutes
set_tests_properties(AtlasIntegration PROPERTIES TIMEOUT 600)   # 10 minutes

# CUDA tests if available
if(CUDA_FOUND)
    add_test(NAME AtlasCUDAPerformance 
             COMMAND test-atlas-cuda-performance)
    set_tests_properties(AtlasCUDAPerformance PROPERTIES TIMEOUT 900)  # 15 minutes
endif()

# Print configuration summary
message(STATUS "ATLAS Testing Framework Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "  CUDA Version: ${CUDA_VERSION}")
endif()
message(STATUS "  Test Executables:")
message(STATUS "    - test-atlas-comprehensive")
message(STATUS "    - test-memory-module")  
message(STATUS "    - test-omega-rule")
message(STATUS "    - test-atlas-integration")
message(STATUS "    - test-atlas-benchmark")
if(CUDA_FOUND)
    message(STATUS "    - test-atlas-cuda-performance")
endif()