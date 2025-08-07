## Summary

Implement the foundational infrastructure for ATLAS (Learning to Optimally Memorize the Context at Test Time) integration into llama.cpp. This is the first phase of a comprehensive multi-phase implementation plan.

## Background

ATLAS introduces test-time memory optimization that transforms traditional attention mechanisms through:
- **Deep memory modules**: 2-layer MLPs with residual connections
- **Omega Rule**: Context-aware memory updates via sliding windows  
- **Muon optimizer**: Newton-Schulz iterations for second-order optimization
- **Feature mappings**: Polynomial/exponential kernels for enhanced memory capacity

## Phase 1 Scope

### 1. Project Structure Setup
- [ ] Create `src/atlas/` directory structure
- [ ] Add core header files: `atlas-core.h`, `memory-module.h`, `omega-rule.h`
- [ ] Implement basic data structures and interfaces
- [ ] Add CUDA-specific backend files

### 2. GGML Extensions
- [ ] Add new GGML operations for ATLAS components:
  - `GGML_OP_ATLAS_MEMORY_UPDATE`
  - `GGML_OP_ATLAS_OMEGA_RULE` 
  - `GGML_OP_ATLAS_MUON_STEP`
  - `GGML_OP_ATLAS_POLY_KERNEL`
- [ ] Extend tensor type system for ATLAS-specific tensors
- [ ] Update operation dispatch in ggml backends

### 3. Core Data Structures
- [ ] Implement `atlas_context` structure
- [ ] Create `atlas_memory_module` with MLP components
- [ ] Add sliding window buffer management
- [ ] Implement momentum state for Muon optimizer

### 4. Build System Integration
- [ ] Add `GGML_USE_ATLAS` CMake option
- [ ] Configure conditional compilation flags
- [ ] Update CUDA backend build configuration
- [ ] Add ATLAS-specific compiler flags and optimizations

### 5. Basic Memory Management
- [ ] Extend `ggml_gallocr` for persistent ATLAS memory
- [ ] Implement hybrid allocation strategy
- [ ] Add circular buffer support for sliding windows
- [ ] Memory lifecycle management for ATLAS components

## Implementation Details

### Directory Structure
```
src/atlas/
├── atlas-core.h/cpp         # Core ATLAS implementation
├── memory-module.h/cpp      # Deep memory MLP with residuals  
├── omega-rule.h/cpp         # Omega Rule sliding window updates
├── muon-optimizer.h/cpp     # Muon optimizer with Newton-Schulz
├── feature-mapping.h/cpp    # Polynomial/exponential kernels
└── cuda/
    ├── atlas-cuda.h/cpp     # CUDA backend implementation
    └── kernels/             # CUDA kernel implementations
```

### Key Data Structures
```c
struct atlas_context {
    struct atlas_memory_module memory;
    struct ggml_tensor * momentum_state;
    struct ggml_tensor * context_window; 
    int window_size;
    float omega_alpha;
    int muon_iterations;
    enum atlas_kernel_type kernel_type;
};

struct atlas_memory_module {
    struct ggml_tensor * w1, * b1;  // First layer
    struct ggml_tensor * w2, * b2;  // Second layer  
    struct ggml_tensor * residual;  // Residual connection
    int input_dim, hidden_dim;
};
```

## Testing Strategy
- [ ] Unit tests for core data structures
- [ ] Memory allocation/deallocation tests
- [ ] CUDA kernel compilation tests
- [ ] Basic functionality validation

## Success Criteria
- [ ] Clean compilation with `-DGGML_USE_ATLAS=ON`
- [ ] All unit tests pass
- [ ] Memory management works without leaks
- [ ] CUDA backend compiles and initializes correctly
- [ ] Foundation ready for Phase 2 component implementation

## Dependencies
- CUDA Toolkit (for CUDA backend)
- CMake 3.18+
- C++17 compiler support

## Estimated Effort
**2 weeks** for experienced C++/CUDA developer

## Related Issues
This is part of a multi-phase ATLAS integration plan:
- Phase 1: Core Infrastructure (this issue)
- Phase 2: Component Implementation (next)
- Phase 3: llama.cpp Integration
- Phase 4: CUDA Optimization
- Phase 5: Testing & Validation

## References
- [ATLAS Paper](https://arxiv.org/abs/2505.23735)
- [llama.cpp Architecture](https://github.com/ggml-org/llama.cpp)
- [GGML Tensor Library](https://github.com/ggerganov/ggml)
