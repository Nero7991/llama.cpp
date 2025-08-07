# ATLAS Testing Framework

This document describes the comprehensive testing framework for the ATLAS (Advanced Tensor Learning and Attention System) Phase 1 implementation in llama.cpp.

## Overview

The ATLAS testing framework provides comprehensive validation of all ATLAS components, ensuring correctness, performance, and robustness across different backends and configurations.

## Test Structure

### Core Test Files

1. **`test-atlas-types.cpp`** - Tests core ATLAS data structures and types
2. **`test-atlas-backend.cpp`** - Tests backend registry and operation support
3. **`test-atlas-memory.cpp`** - Tests memory allocation and pool management
4. **`test-atlas-operations.cpp`** - Tests GGML ATLAS operations integration
5. **`test-atlas-integration.cpp`** - Comprehensive end-to-end integration tests

### Test Categories

#### 1. Types and Data Structures (`test-atlas-types.cpp`)
- **Memory Module Creation** - Validates ATLAS memory module initialization
- **Context Management** - Tests ATLAS context configuration and state
- **Tensor Descriptors** - Verifies tensor descriptor management
- **Memory Pool Management** - Tests memory pool allocation and tracking
- **Feature Mapping Types** - Validates kernel types and activation functions
- **Multi-threaded Access** - Tests thread-safe memory operations
- **Memory Alignment** - Ensures proper memory alignment requirements

#### 2. Backend Integration (`test-atlas-backend.cpp`)
- **Backend Registration** - Tests backend discovery and registration
- **Operation Support** - Validates operation compatibility checking
- **Memory Management** - Tests backend-specific memory allocation
- **Performance Metrics** - Verifies performance measurement capabilities
- **Backend Selection** - Tests automatic backend selection strategies
- **Synchronization** - Validates backend synchronization mechanisms

#### 3. Memory Management (`test-atlas-memory.cpp`)
- **Pool Creation/Destruction** - Tests memory pool lifecycle
- **Allocation/Deallocation** - Validates memory allocation strategies
- **Alignment Requirements** - Tests various alignment requirements
- **Fragmentation/Coalescing** - Tests memory fragmentation handling
- **Multi-threaded Access** - Validates thread-safe memory operations
- **Statistics Tracking** - Tests memory usage statistics
- **Multiple Memory Types** - Tests host/device/pinned memory types
- **Allocation Strategies** - Compares first-fit/best-fit/worst-fit strategies

#### 4. ATLAS Operations (`test-atlas-operations.cpp`)
- **Memory Module Forward** - Tests deep memory module computation
- **Omega Rule Updates** - Validates sliding window memory updates
- **Newton-Schulz Inverse** - Tests matrix inverse approximation
- **Muon Optimizer** - Validates second-order optimization steps
- **Feature Mapping** - Tests polynomial/exponential/RBF kernels
- **Graph Construction** - Tests GGML graph building with ATLAS ops
- **Numerical Stability** - Validates numerical robustness

#### 5. Integration Testing (`test-atlas-integration.cpp`)
- **End-to-End Pipeline** - Tests complete ATLAS workflow
- **Memory Window Management** - Validates sliding window operations
- **Multi-batch Processing** - Tests batch processing capabilities
- **Memory Performance** - Validates memory usage and performance
- **Gradient Flow** - Tests gradient computation and optimization
- **CUDA Integration** - Tests CUDA backend integration (if available)

## Building and Running Tests

### Prerequisites

- CMake 3.14 or higher
- C++17 compatible compiler
- GGML and llama.cpp dependencies
- Optional: CUDA toolkit for GPU tests

### Building ATLAS Tests

```bash
# Configure build with ATLAS tests enabled
cmake -B build -DGGML_BACKEND_DL=OFF

# Build all ATLAS tests
cmake --build build --target test-atlas-types
cmake --build build --target test-atlas-backend
cmake --build build --target test-atlas-memory
cmake --build build --target test-atlas-operations
cmake --build build --target test-atlas-integration

# Or build all tests at once
cmake --build build -j$(nproc)
```

### Running Tests

#### Using the Test Runner (Recommended)

```bash
# Run all ATLAS tests
./tests/run-atlas-tests.sh

# Run with verbose output
./tests/run-atlas-tests.sh --verbose

# Run CPU-only tests
./tests/run-atlas-tests.sh --cpu-only

# Run quick test subset
./tests/run-atlas-tests.sh --quick

# Run specific test
./tests/run-atlas-tests.sh --test test-atlas-memory

# Check test dependencies
./tests/run-atlas-tests.sh --check-deps

# Run performance benchmarks
./tests/run-atlas-tests.sh --benchmark

# List available tests
./tests/run-atlas-tests.sh --list-tests
```

#### Using CTest

```bash
# Run all ATLAS tests via CTest
cd build
ctest -L atlas

# Run with verbose output
ctest -L atlas -V

# Run specific test
ctest -R test-atlas-memory
```

#### Running Individual Tests

```bash
# From build directory
cd build/bin

# Run individual tests
./test-atlas-types
./test-atlas-backend
./test-atlas-memory
./test-atlas-operations
./test-atlas-integration
```

## Test Configuration

### CMake Options

The ATLAS tests support various CMake configuration options:

```cmake
# Enable CUDA support for ATLAS tests
-DLLAMA_CUDA=ON

# Enable Metal support for ATLAS tests  
-DLLAMA_METAL=ON

# Enable OpenCL support for ATLAS tests
-DLLAMA_OPENCL=ON

# Disable dynamic loading (required for ATLAS tests)
-DGGML_BACKEND_DL=OFF
```

### Environment Variables

```bash
# Set specific GPU device for CUDA tests
export CUDA_VISIBLE_DEVICES=0

# Enable additional debugging output
export ATLAS_DEBUG=1

# Set memory limits for tests
export ATLAS_TEST_MEMORY_LIMIT=1073741824  # 1GB
```

## Backend-Specific Testing

### CPU Testing

CPU tests are always enabled and test:
- Basic ATLAS operations on CPU
- Memory management with host memory
- Single-threaded and multi-threaded scenarios
- Various memory alignment requirements

### CUDA Testing

CUDA tests are enabled when `LLAMA_CUDA=ON` and test:
- GPU memory allocation and management
- CUDA kernel execution for ATLAS operations
- Host-device memory transfers
- CUDA-specific optimizations
- Mixed precision operations

### Metal Testing

Metal tests are enabled when `LLAMA_METAL=ON` and test:
- Metal compute shader execution
- Metal memory management
- Apple Silicon optimizations

### OpenCL Testing

OpenCL tests are enabled when `LLAMA_OPENCL=ON` and test:
- OpenCL kernel execution
- Cross-platform GPU compatibility
- OpenCL memory management

## Performance Testing

### Benchmarking

The test framework includes performance benchmarking capabilities:

```bash
# Run performance benchmarks
./tests/run-atlas-tests.sh --benchmark
```

Benchmarks measure:
- Memory allocation/deallocation speed
- ATLAS operation execution times
- Memory bandwidth utilization
- Throughput for various batch sizes
- Scaling across multiple GPUs/cores

### Memory Usage Analysis

Tests monitor memory usage and report:
- Peak memory consumption
- Memory fragmentation levels
- Memory pool utilization ratios
- Memory bandwidth efficiency

### Performance Regression Detection

The framework can detect performance regressions by:
- Comparing execution times against baselines
- Monitoring memory usage patterns
- Tracking throughput variations
- Validating numerical accuracy

## Test Data and Fixtures

### Test Data Generation

Tests use deterministic random number generation for reproducibility:

```cpp
// Seeded random generation for consistent test data
std::random_device rd;
std::mt19937 gen(12345);  // Fixed seed for reproducibility
std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
```

### Test Configurations

Common test configurations are defined in:

```cpp
struct atlas_test_config {
    int sequence_length;      // Default: 1024
    int batch_size;          // Default: 4
    int hidden_dimension;    // Default: 768
    int memory_window_size;  // Default: 256
    float learning_rate;     // Default: 0.001f
    int muon_iterations;     // Default: 5
    bool use_cuda;          // Default: false
    float tolerance;        // Default: 1e-3f
};
```

## Debugging and Troubleshooting

### Common Issues

1. **Memory Allocation Failures**
   - Check available system memory
   - Reduce test batch sizes or sequence lengths
   - Verify memory alignment requirements

2. **CUDA Test Failures**
   - Verify CUDA installation and driver version
   - Check GPU memory availability
   - Ensure CUDA compute capability compatibility

3. **Numerical Precision Issues**
   - Adjust tolerance values for different backends
   - Check for numerical instability in operations
   - Verify proper handling of edge cases

### Debug Output

Enable detailed debug output:

```bash
# Environment variable
export ATLAS_DEBUG=1

# Or use verbose test runner
./tests/run-atlas-tests.sh --verbose
```

### Memory Debugging

For memory-related issues:

```bash
# Run with Valgrind
valgrind --tool=memcheck ./build/bin/test-atlas-memory

# Run with AddressSanitizer
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON
cmake --build build --target test-atlas-memory
./build/bin/test-atlas-memory
```

## Test Coverage

### Code Coverage Analysis

Generate code coverage reports:

```bash
# Configure with coverage enabled
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON

# Build and run tests
cmake --build build
cd build
ctest -L atlas

# Generate coverage report
gcov -r ../tests/test-atlas-*.cpp
lcov --directory . --capture --output-file coverage.info
genhtml coverage.info --output-directory coverage_html
```

### Coverage Goals

Target coverage levels:
- **Line Coverage**: > 90%
- **Branch Coverage**: > 85%
- **Function Coverage**: > 95%

## Continuous Integration

### CI Pipeline Integration

The ATLAS tests integrate with CI systems:

```yaml
# GitHub Actions example
- name: Run ATLAS Tests
  run: |
    cmake -B build -DGGML_BACKEND_DL=OFF
    cmake --build build -j$(nproc)
    ./tests/run-atlas-tests.sh --cpu-only
```

### Test Matrix

CI runs tests across:
- Multiple OS platforms (Linux, macOS, Windows)
- Different compiler versions (GCC, Clang, MSVC)
- Various backend configurations (CPU, CUDA, Metal, OpenCL)
- Different build configurations (Debug, Release, RelWithDebInfo)

## Contributing to Tests

### Adding New Tests

1. Create test functions following existing patterns
2. Add comprehensive error checking and validation
3. Include performance benchmarking where relevant
4. Document test purpose and expected behavior
5. Add test to appropriate CMake configuration

### Test Guidelines

1. **Deterministic**: Tests should produce consistent results
2. **Independent**: Tests should not depend on other test outcomes
3. **Comprehensive**: Cover both success and failure scenarios
4. **Performance-aware**: Include timing and memory measurements
5. **Well-documented**: Clear comments explaining test purpose

### Review Process

All test additions should:
1. Pass all existing tests
2. Include appropriate documentation
3. Follow coding standards
4. Include performance validation
5. Be reviewed by ATLAS maintainers

## Future Enhancements

### Planned Test Additions

1. **Stress Testing**: Extended duration and high-load tests
2. **Fuzzing**: Automated input fuzzing for robustness
3. **Distributed Testing**: Multi-GPU and multi-node testing
4. **Model Validation**: End-to-end model inference testing
5. **Backward Compatibility**: Regression testing across versions

### Test Automation Improvements

1. **Automated Benchmarking**: Continuous performance monitoring
2. **Test Generation**: Automated test case generation
3. **Coverage Analysis**: Automated coverage reporting
4. **Performance Regression**: Automated performance regression detection

---

This testing framework ensures the ATLAS implementation meets the highest standards for correctness, performance, and reliability across all supported configurations and backends.