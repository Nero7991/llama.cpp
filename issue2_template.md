## Summary

Implement the core deep memory module for ATLAS - a 2-layer MLP with residual connections that replaces traditional shallow key-value caches. This module forms the foundation of ATLAS's enhanced memory capacity.

## Background

The deep memory module transforms simple linear projections into learnable non-linear representations:
- **Traditional**: `K = X * W_k`, `V = X * W_v` (shallow projections)
- **ATLAS**: 2-layer MLP with residual connections and activation functions

## Implementation Requirements

### 1. Core Data Structure
```c
struct atlas_memory_module {
    // First layer: input_dim → hidden_dim
    struct ggml_tensor * w1;        // Weight matrix [input_dim, hidden_dim]
    struct ggml_tensor * b1;        // Bias vector [hidden_dim]
    
    // Second layer: hidden_dim → output_dim  
    struct ggml_tensor * w2;        // Weight matrix [hidden_dim, output_dim]
    struct ggml_tensor * b2;        // Bias vector [output_dim]
    
    // Residual connection: input_dim → output_dim
    struct ggml_tensor * w_residual; // Weight matrix [input_dim, output_dim]
    
    // Configuration
    int input_dim;                   // Input dimension (typically model dim)
    int hidden_dim;                  // Hidden layer dimension (2x-4x input_dim)
    int output_dim;                  // Output dimension (typically model dim)
    enum atlas_activation_type activation; // GELU, ReLU, etc.
};
```

### 2. Forward Pass Implementation
```c
struct ggml_tensor * atlas_memory_forward(
    struct ggml_context * ctx,
    struct atlas_memory_module * module,
    struct ggml_tensor * input);
```

**Algorithm**:
1. First layer: `h1 = activation(input * W1 + b1)`
2. Second layer: `h2 = h1 * W2 + b2`  
3. Residual: `output = h2 + input * W_residual`

### 3. CUDA Kernel Implementation
```cuda
__global__ void atlas_memory_forward_kernel(
    const float* input,      // [batch_size, input_dim]
    const float* w1,         // [input_dim, hidden_dim]
    const float* b1,         // [hidden_dim] 
    const float* w2,         // [hidden_dim, output_dim]
    const float* b2,         // [output_dim]
    const float* w_residual, // [input_dim, output_dim]
    float* output,           // [batch_size, output_dim]
    int batch_size, 
    int input_dim, 
    int hidden_dim, 
    int output_dim,
    int activation_type
);
```

**Optimization Requirements**:
- Memory coalescing for optimal bandwidth utilization
- Shared memory usage for frequently accessed data
- Thread block optimization for different tensor sizes
- Support for mixed precision (FP16/FP32)

### 4. Activation Functions
Support for multiple activation functions:
- **GELU**: `x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
- **ReLU**: `max(0, x)`
- **SiLU/Swish**: `x * sigmoid(x)`

### 5. Memory Module Initialization
```c
int atlas_memory_module_init(
    struct atlas_memory_module * module,
    struct ggml_context * ctx,
    int input_dim,
    int hidden_dim, 
    int output_dim,
    enum atlas_activation_type activation
);
```

**Initialization Strategy**:
- Xavier/Glorot initialization for weights
- Zero initialization for biases
- Proper tensor allocation and lifetime management

## Testing Requirements

### Unit Tests
- [ ] **Forward pass correctness**: Compare against reference PyTorch implementation
- [ ] **Gradient computation**: Verify backward pass for training scenarios
- [ ] **Memory allocation**: Test proper tensor allocation/deallocation
- [ ] **Activation functions**: Validate all supported activation types
- [ ] **Edge cases**: Handle zero/negative inputs, extreme values
- [ ] **Dimension compatibility**: Test various input/hidden/output dimensions

### Performance Tests  
- [ ] **CUDA kernel benchmarks**: Compare against cuBLAS implementations
- [ ] **Memory bandwidth utilization**: Achieve >80% theoretical bandwidth
- [ ] **Scalability testing**: Test with different batch sizes and dimensions
- [ ] **Mixed precision**: Validate FP16 accuracy vs FP32

### Integration Tests
- [ ] **GGML integration**: Verify seamless operation with GGML tensor system
- [ ] **Memory management**: Test with ggml_gallocr allocation patterns
- [ ] **Backend compatibility**: Ensure CPU fallback works correctly

## Implementation Files

### Core Implementation
- `src/atlas/memory-module.h` - Header with structure definitions
- `src/atlas/memory-module.cpp` - CPU implementation
- `src/atlas/cuda/memory-module-cuda.cu` - CUDA implementation
- `src/atlas/cuda/memory-kernels.cuh` - CUDA kernel declarations

### Test Files
- `tests/atlas/test-memory-module.cpp` - Unit tests
- `tests/atlas/test-memory-cuda.cu` - CUDA-specific tests
- `tests/atlas/benchmark-memory.cpp` - Performance benchmarks

## Success Criteria

### Functional Requirements
- [ ] Forward pass produces mathematically correct results
- [ ] All activation functions work correctly
- [ ] Memory allocation/deallocation without leaks
- [ ] CUDA implementation matches CPU results (within numerical precision)
- [ ] Integration with GGML tensor operations

### Performance Requirements
- [ ] CUDA implementation achieves >80% of theoretical peak performance
- [ ] Memory usage scales linearly with batch size
- [ ] Latency overhead <10% compared to simple matrix multiplication
- [ ] Support for tensors up to 32K context length

### Quality Requirements
- [ ] 100% unit test coverage
- [ ] All tests pass on multiple GPU architectures (Ampere, Ada)
- [ ] Clean compilation with -Wall -Wextra
- [ ] Memory leak detection passes (valgrind/CUDA memcheck)

## Dependencies
- CUDA Toolkit 11.0+
- cuBLAS library
- GGML tensor system
- C++17 compiler

## Integration Points
- **Omega Rule**: Memory module output feeds into sliding window updates
- **Muon Optimizer**: Gradients computed through this module for optimization
- **Feature Mapping**: Input preprocessing before memory module
- **Attention**: Enhanced attention uses memory module outputs

## Performance Targets
- **Throughput**: Process 4096-dimensional vectors at >1M vectors/sec
- **Memory**: <50MB additional overhead for typical configurations  
- **Latency**: <2ms for single forward pass on RTX 4090
- **Accuracy**: Numerical results within 1e-5 of reference implementation

## References
- ATLAS Paper Section 3.1: "Deep Memory Modules"
- GGML Documentation: Tensor Operations
- CUDA Best Practices Guide: Memory Optimization
