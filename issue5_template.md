## Summary

Implement the Muon optimizer with Newton-Schulz iterations for ATLAS - providing second-order optimization for memory module parameters. This component enables faster convergence and better optimization compared to first-order methods.

## Background

The Muon optimizer enhances traditional gradient descent with second-order information:
- **Traditional Optimizers**: First-order methods (SGD, Adam) using only gradients
- **Muon**: Second-order method using approximate Hessian via Newton-Schulz iterations

**Mathematical Foundation**:
```
θ_{t+1} = θ_t - α * H^{-1} * ∇L(θ_t)
```
Where `H^{-1}` is approximated using Newton-Schulz iterations instead of expensive matrix inversion.

## Implementation Requirements

### 1. Core Data Structures
```c
struct atlas_muon_optimizer {
    // Momentum state
    struct ggml_tensor * momentum_buffer;    // [param_count] momentum values
    struct ggml_tensor * hessian_approx;     // [param_count, param_count] Hessian approximation
    
    // Newton-Schulz iteration workspace
    struct ggml_tensor * ns_workspace;       // [param_count, param_count] temporary matrices
    struct ggml_tensor * ns_temp1;           // [param_count, param_count] 
    struct ggml_tensor * ns_temp2;           // [param_count, param_count]
    
    // Optimizer parameters
    float learning_rate;                     // α in update equation
    float momentum_decay;                    // β for momentum (typically 0.9)
    float weight_decay;                      // L2 regularization strength
    int newton_schulz_iterations;            // Number of NS iterations (3-5)
    float hessian_update_frequency;          // Update Hessian every N steps
    
    // State tracking
    int step_count;                          // Current optimization step
    float last_loss;                         // Previous loss for convergence monitoring
    bool use_adaptive_lr;                    // Adaptive learning rate based on progress
};
```

### 2. Newton-Schulz Matrix Inversion
```c
// Newton-Schulz iterative matrix inversion: X_{k+1} = X_k * (3*I - A*X_k) / 2
struct ggml_tensor * newton_schulz_inverse(
    struct ggml_context * ctx,
    const struct ggml_tensor * matrix,
    int max_iterations,
    float convergence_tolerance,
    struct ggml_tensor * workspace
);

// Single Newton-Schulz iteration
struct ggml_tensor * newton_schulz_step(
    struct ggml_context * ctx,
    const struct ggml_tensor * current_inverse,
    const struct ggml_tensor * original_matrix,
    struct ggml_tensor * temp_buffer
);
```

**Algorithm**:
1. Initialize: `X_0 = α * A^T` where `α = 1/||A||_F^2`
2. Iterate: `X_{k+1} = X_k * (3*I - A*X_k) / 2`
3. Convergence check: `||A*X_k - I||_F < tolerance`

### 3. Muon Optimization Step
```c
// Complete Muon optimization step
int atlas_muon_step(
    struct atlas_muon_optimizer * optimizer,
    struct ggml_tensor * parameters,
    const struct ggml_tensor * gradients,
    struct ggml_context * ctx
);

// Hessian approximation update
void atlas_muon_update_hessian(
    struct atlas_muon_optimizer * optimizer,
    const struct ggml_tensor * gradients,
    const struct ggml_tensor * prev_gradients,
    struct ggml_context * ctx
);
```

**Algorithm**:
1. Compute Hessian approximation (BFGS-style or L-BFGS)
2. Apply Newton-Schulz to get `H^{-1}`
3. Compute update: `delta = H^{-1} * gradients`
4. Apply momentum: `m_t = β*m_{t-1} + (1-β)*delta`
5. Update parameters: `θ_{t+1} = θ_t - α*m_t`
6. Apply weight decay if enabled

### 4. CUDA Implementation
```cuda
// Newton-Schulz iteration kernel
__global__ void newton_schulz_iteration_kernel(
    const float* matrix_A,      // [dim, dim] original matrix
    const float* current_X,     // [dim, dim] current inverse approximation
    float* next_X,              // [dim, dim] next iteration result
    float* workspace,           // [dim, dim] temporary storage
    int dim,
    float damping_factor        // For numerical stability
);

// Muon parameter update kernel
__global__ void muon_update_kernel(
    float* parameters,          // [param_count] model parameters
    const float* gradients,     // [param_count] current gradients
    const float* hessian_inv,   // [param_count, param_count] inverse Hessian
    float* momentum,            // [param_count] momentum buffer
    int param_count,
    float learning_rate,
    float momentum_decay,
    float weight_decay
);

// Hessian approximation update kernel (BFGS-style)
__global__ void hessian_update_kernel(
    float* hessian,             // [param_count, param_count] Hessian approximation
    const float* grad_diff,     // [param_count] gradient difference
    const float* param_diff,    // [param_count] parameter difference
    int param_count,
    float damping              // Regularization for numerical stability
);
```

### 5. Adaptive Learning Rate
```c
// Adaptive learning rate based on optimization progress
float atlas_muon_adaptive_lr(
    const struct atlas_muon_optimizer * optimizer,
    float current_loss,
    float previous_loss
);

// Convergence detection
bool atlas_muon_check_convergence(
    const struct atlas_muon_optimizer * optimizer,
    float loss_threshold,
    int patience_steps
);
```

### 6. Memory-Efficient Implementation
```c
// Low-rank Hessian approximation for large parameter spaces
struct atlas_muon_lowrank {
    struct ggml_tensor * U;     // [param_count, rank] left factors
    struct ggml_tensor * V;     // [rank, param_count] right factors  
    struct ggml_tensor * D;     // [rank] diagonal values
    int rank;                   // Approximation rank (typically 50-200)
};
```

## Testing Requirements

### Unit Tests
- [ ] **Newton-Schulz convergence**: Verify matrix inversion accuracy
- [ ] **Numerical stability**: Test with ill-conditioned matrices
- [ ] **Muon step correctness**: Compare against reference PyTorch implementation
- [ ] **Momentum handling**: Verify momentum accumulation and decay
- [ ] **Hessian approximation**: Test BFGS-style updates
- [ ] **Adaptive learning rate**: Validate LR adjustment logic

### Performance Tests
- [ ] **Newton-Schulz efficiency**: Benchmark iteration count vs accuracy
- [ ] **CUDA optimization**: Achieve >70% memory bandwidth utilization
- [ ] **Large parameter spaces**: Test with 7B+ parameter models
- [ ] **Convergence speed**: Compare convergence rate vs Adam/SGD

### Integration Tests
- [ ] **Memory module integration**: End-to-end optimization with deep memory
- [ ] **Omega Rule integration**: Joint optimization with sliding window loss
- [ ] **Numerical precision**: FP16/FP32 mixed precision testing
- [ ] **Memory management**: No memory leaks during long optimization runs

## Implementation Files

### Core Implementation
- `src/atlas/muon-optimizer.h` - Header with structure definitions
- `src/atlas/muon-optimizer.cpp` - CPU implementation
- `src/atlas/newton-schulz.cpp` - Newton-Schulz matrix inversion
- `src/atlas/cuda/muon-cuda.cu` - CUDA implementation
- `src/atlas/cuda/newton-schulz-kernels.cu` - Newton-Schulz CUDA kernels

### Low-Rank Optimization
- `src/atlas/muon-lowrank.cpp` - Memory-efficient low-rank approximation
- `src/atlas/cuda/lowrank-kernels.cu` - CUDA kernels for low-rank operations

### Test Files
- `tests/atlas/test-muon-optimizer.cpp` - Core functionality tests
- `tests/atlas/test-newton-schulz.cpp` - Matrix inversion tests
- `tests/atlas/benchmark-muon.cpp` - Performance benchmarks

## Success Criteria

### Functional Requirements
- [ ] Newton-Schulz achieves matrix inversion within 1e-6 accuracy
- [ ] Muon optimizer converges faster than Adam baseline (measured in steps)
- [ ] Numerical stability maintained for condition numbers up to 1e8
- [ ] Memory usage scales reasonably with parameter count

### Performance Requirements
- [ ] Newton-Schulz converges in 3-5 iterations for well-conditioned matrices
- [ ] CUDA implementation achieves >70% theoretical memory bandwidth
- [ ] Overall optimization overhead <20% compared to first-order methods
- [ ] Support for parameter spaces up to 70B parameters (with low-rank approximation)

### Quality Requirements
- [ ] 100% unit test coverage for all optimization components
- [ ] Convergence tests on standard optimization benchmarks
- [ ] Memory leak testing for extended optimization runs
- [ ] Cross-platform compatibility (CUDA, CPU fallback)

## Advanced Features

### 1. Preconditioning
- Diagonal preconditioning for improved conditioning
- Block-diagonal approximations for computational efficiency

### 2. Regularization
- Tikhonov regularization for numerical stability
- Adaptive regularization based on condition number

### 3. Distributed Optimization
- Support for model parallel optimization
- Gradient synchronization across devices

## Dependencies
- Issue #3: Deep Memory Module (for parameter optimization)
- Issue #4: Omega Rule (for loss computation)
- CUDA Toolkit 11.0+ with cuBLAS/cuSOLVER
- Linear algebra libraries (BLAS, LAPACK)

## Performance Targets
- **Convergence**: 2-3x faster than Adam optimizer
- **Memory**: <2GB additional overhead for 7B parameter model
- **Throughput**: Process optimization steps at >10 steps/second
- **Numerical accuracy**: Matrix inversion within 1e-6 relative error

## Estimated Effort
**2-3 weeks** for experienced CUDA/optimization developer

## References
- ATLAS Paper Section 3.3: "Muon Optimizer"
- Newton-Schulz iteration: Convergence analysis
- BFGS and L-BFGS optimization methods
- Second-order optimization in deep learning
