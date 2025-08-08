## Summary

Implement feature mapping components for ATLAS - polynomial and exponential kernels that enhance memory capacity beyond traditional matrix storage limitations. This component provides the foundation for ATLAS's expanded representational power.

## Background

Feature mapping transforms input vectors into higher-dimensional feature spaces without explicitly storing the expanded representations:
- **Traditional**: Direct key-value storage with limited capacity
- **ATLAS**: Rich feature spaces through kernel methods enabling exponential capacity increase

**Mathematical Foundation**:
```
φ_poly(x) = [x, x², x³, ..., x^p]           # Polynomial features
φ_exp(x) = exp(x ⊙ W_proj)                  # Exponential features
φ_fourier(x) = [sin(πx), cos(πx), ...]      # Fourier features
```

## Implementation Requirements

### 1. Core Data Structures
```c
enum atlas_feature_type {
    ATLAS_FEATURE_IDENTITY,      // φ(x) = x (baseline)
    ATLAS_FEATURE_POLYNOMIAL,    // φ(x) = [x, x², x³, ...]
    ATLAS_FEATURE_EXPONENTIAL,   // φ(x) = exp(x ⊙ W)
    ATLAS_FEATURE_FOURIER,       // φ(x) = [sin(πx), cos(πx), ...]
    ATLAS_FEATURE_RBF,           // φ(x) = exp(-||x-c||²/σ²)
    ATLAS_FEATURE_RANDOM         // φ(x) = random projection
};

struct atlas_feature_params {
    enum atlas_feature_type type;
    
    // Polynomial parameters
    int polynomial_degree;           // Degree p for x^p terms
    bool include_cross_terms;        // Include x_i * x_j terms
    
    // Exponential parameters  
    struct ggml_tensor * exp_weights; // [input_dim, proj_dim] projection weights
    int projection_dim;              // Dimensionality of projection
    float temperature;               // Temperature for exp scaling
    
    // Fourier parameters
    int num_frequencies;             // Number of frequency components
    float frequency_scale;           // Scaling factor for frequencies
    struct ggml_tensor * freq_matrix; // [input_dim, num_freq] frequency matrix
    
    // RBF parameters
    struct ggml_tensor * rbf_centers; // [num_centers, input_dim] RBF centers
    float rbf_bandwidth;             // σ² in RBF kernel
    int num_centers;                 // Number of RBF centers
    
    // Random projection parameters
    struct ggml_tensor * random_matrix; // [input_dim, output_dim] random projection
    int output_dimension;            // Target output dimensionality
    bool normalize_features;         // L2 normalize output features
};

struct atlas_feature_mapper {
    struct atlas_feature_params params;
    struct ggml_tensor * feature_cache;    // [cache_size, feature_dim] cached features
    struct ggml_tensor * input_cache;      // [cache_size, input_dim] cached inputs
    int cache_size;                        // Number of cached entries
    int cache_head;                        // Current cache position
    bool enable_caching;                   // Enable feature caching
};
```

### 2. Core Feature Mapping Functions
```c
// Main feature mapping interface
struct ggml_tensor * atlas_apply_feature_mapping(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,           // [batch_size, input_dim]
    const struct atlas_feature_mapper * mapper
);

// Individual feature mapping implementations
struct ggml_tensor * atlas_polynomial_features(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,
    int degree,
    bool include_cross_terms
);

struct ggml_tensor * atlas_exponential_features(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,
    const struct ggml_tensor * projection_weights,
    float temperature
);

struct ggml_tensor * atlas_fourier_features(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,
    const struct ggml_tensor * frequency_matrix,
    float frequency_scale
);

struct ggml_tensor * atlas_rbf_features(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,
    const struct ggml_tensor * centers,
    float bandwidth
);
```

### 3. CUDA Implementations
```cuda
// Polynomial feature computation kernel
__global__ void atlas_polynomial_kernel(
    const float* input,         // [batch_size, input_dim]
    float* output,              // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int degree,
    bool include_cross_terms
);

// Exponential feature computation kernel  
__global__ void atlas_exponential_kernel(
    const float* input,         // [batch_size, input_dim]
    const float* proj_weights,  // [input_dim, proj_dim]
    float* output,              // [batch_size, proj_dim]
    int batch_size,
    int input_dim,
    int proj_dim,
    float temperature
);

// Fourier feature computation kernel
__global__ void atlas_fourier_kernel(
    const float* input,         // [batch_size, input_dim]
    const float* freq_matrix,   // [input_dim, num_freq]
    float* output,              // [batch_size, 2*num_freq] (sin + cos)
    int batch_size,
    int input_dim,
    int num_frequencies,
    float frequency_scale
);

// RBF feature computation kernel
__global__ void atlas_rbf_kernel(
    const float* input,         // [batch_size, input_dim]
    const float* centers,       // [num_centers, input_dim]
    float* output,              // [batch_size, num_centers]
    int batch_size,
    int input_dim,
    int num_centers,
    float bandwidth
);
```

### 4. Feature Dimension Calculation
```c
// Calculate output dimension for different feature types
int atlas_calculate_feature_dimension(
    int input_dim,
    const struct atlas_feature_params * params
);

// Specific dimension calculators
int atlas_polynomial_output_dim(int input_dim, int degree, bool cross_terms);
int atlas_exponential_output_dim(int projection_dim);
int atlas_fourier_output_dim(int num_frequencies);  // 2 * num_frequencies
int atlas_rbf_output_dim(int num_centers);
```

### 5. Feature Caching System
```c
// Initialize feature cache for frequently used inputs
int atlas_feature_cache_init(
    struct atlas_feature_mapper * mapper,
    int cache_size,
    int input_dim,
    int feature_dim
);

// Check cache for precomputed features
struct ggml_tensor * atlas_feature_cache_lookup(
    const struct atlas_feature_mapper * mapper,
    const struct ggml_tensor * input,
    float similarity_threshold
);

// Add computed features to cache
void atlas_feature_cache_add(
    struct atlas_feature_mapper * mapper,
    const struct ggml_tensor * input,
    const struct ggml_tensor * features
);
```

### 6. Adaptive Feature Selection
```c
// Select optimal feature mapping based on data characteristics
enum atlas_feature_type atlas_select_optimal_features(
    const struct ggml_tensor * sample_data,
    const struct atlas_feature_params * candidates,
    int num_candidates,
    float complexity_budget
);

// Estimate computational cost of feature mapping
float atlas_estimate_feature_cost(
    int input_dim,
    int batch_size,
    const struct atlas_feature_params * params
);
```

## Testing Requirements

### Unit Tests
- [ ] **Polynomial features**: Verify correct polynomial expansion up to degree 5
- [ ] **Exponential features**: Test exponential computation and numerical stability
- [ ] **Fourier features**: Validate sin/cos computation and frequency scaling
- [ ] **RBF features**: Test Gaussian RBF computation with various bandwidths
- [ ] **Dimension calculation**: Verify output dimensions for all feature types
- [ ] **Feature caching**: Test cache hit/miss behavior and memory management

### Mathematical Validation
- [ ] **Kernel properties**: Verify positive semi-definiteness where applicable
- [ ] **Numerical precision**: Test with FP16/FP32 mixed precision
- [ ] **Orthogonality**: Validate orthogonal features for Fourier mappings
- [ ] **Normalization**: Test L2 normalization when enabled

### Performance Tests
- [ ] **CUDA optimization**: Achieve >85% memory bandwidth utilization
- [ ] **Scalability**: Test with input dimensions up to 4096
- [ ] **Batch processing**: Efficient handling of large batch sizes
- [ ] **Cache efficiency**: Measure cache hit rates and speedup

### Integration Tests
- [ ] **Memory module integration**: Feature mapping → deep memory module pipeline
- [ ] **Omega Rule integration**: Features used in sliding window loss computation
- [ ] **Backend compatibility**: CPU fallback implementations
- [ ] **Memory management**: No leaks during long-running feature computation

## Implementation Files

### Core Implementation
- `src/atlas/feature-mapping.h` - Header with structure definitions and interfaces
- `src/atlas/feature-mapping.cpp` - CPU implementations for all feature types
- `src/atlas/feature-cache.cpp` - Feature caching system implementation
- `src/atlas/feature-selection.cpp` - Adaptive feature selection algorithms

### CUDA Implementation
- `src/atlas/cuda/feature-cuda.cu` - CUDA implementations for all kernels
- `src/atlas/cuda/feature-kernels.cuh` - CUDA kernel declarations
- `src/atlas/cuda/feature-cache-cuda.cu` - GPU-accelerated caching

### Mathematical Utilities
- `src/atlas/math/polynomial.cpp` - Polynomial expansion utilities
- `src/atlas/math/fourier.cpp` - Fourier transform utilities
- `src/atlas/math/rbf.cpp` - RBF kernel computations

### Test Files
- `tests/atlas/test-feature-mapping.cpp` - Core functionality tests
- `tests/atlas/test-feature-math.cpp` - Mathematical validation tests
- `tests/atlas/benchmark-features.cpp` - Performance benchmarks

## Success Criteria

### Functional Requirements
- [ ] All feature mapping types produce mathematically correct results
- [ ] Feature dimensions calculated correctly for all combinations
- [ ] Caching system reduces computation time by >50% for repeated inputs
- [ ] Numerical stability maintained for extreme input values

### Performance Requirements
- [ ] Polynomial features: Process 4K-dim vectors at >500K vectors/sec
- [ ] Exponential features: >80% memory bandwidth utilization on GPU
- [ ] Fourier features: Maintain numerical precision for high frequencies
- [ ] RBF features: Efficient computation for 1000+ centers

### Quality Requirements
- [ ] 100% unit test coverage for all feature types
- [ ] Mathematical validation against reference implementations
- [ ] Memory usage scales linearly with input/output dimensions
- [ ] Cross-platform compatibility (CUDA, CPU)

## Advanced Features

### 1. Learnable Feature Parameters
- Gradient-based optimization of projection weights
- Adaptive bandwidth selection for RBF kernels
- Dynamic frequency selection for Fourier features

### 2. Compositional Features
- Combination of multiple feature types
- Hierarchical feature mappings
- Sequential feature transformations

### 3. Memory-Efficient Implementation
- Streaming computation for large feature spaces
- Low-rank approximations for expensive mappings
- Sparse feature representations

## Dependencies
- Issue #3: Deep Memory Module (consumes feature-mapped inputs)
- Issue #4: Omega Rule (uses features in loss computation)
- CUDA Toolkit 11.0+ with cuBLAS
- Mathematical libraries (FFTW for Fourier features)

## Performance Targets
- **Throughput**: Process 4096-dimensional inputs at >1M vectors/sec
- **Memory**: <100MB overhead for feature computation workspace
- **Latency**: <1ms for single vector feature mapping on RTX 4090
- **Cache efficiency**: >70% cache hit rate for repeated patterns

## Estimated Effort
**2 weeks** for experienced CUDA/mathematical computing developer

## References
- ATLAS Paper Section 3.4: "Feature Mapping"
- Kernel Methods in Machine Learning
- Random Features for Large-Scale Kernel Machines
- Fourier Features for Shift-Invariant Kernels
