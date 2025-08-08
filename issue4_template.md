## Summary

Implement the Omega Rule for ATLAS - context-aware memory updates using sliding windows. This component enables ATLAS to learn from historical context rather than just the current token, providing the core advantage over traditional attention mechanisms.

## Background

The Omega Rule fundamentally changes how memory is updated:
- **Traditional**: Memory updated with single current token
- **ATLAS**: Memory optimized over sliding window of recent context using L2 reconstruction loss

**Mathematical Foundation**:
```
L_omega(θ, t) = Σ(i=max(0,t-w) to t) ||M_θ(φ*(k_i)) - v_i||²
```
Where:
- `w` = sliding window size
- `M_θ` = memory module with parameters θ  
- `φ*` = feature mapping function
- `k_i, v_i` = key-value pairs at position i

## Implementation Requirements

### 1. Core Data Structures
```c
struct atlas_sliding_window {
    struct ggml_tensor * key_buffer;    // [window_size, key_dim] circular buffer
    struct ggml_tensor * value_buffer;  // [window_size, value_dim] circular buffer
    struct ggml_tensor * position_mask; // [window_size] validity mask
    
    int window_size;                    // Maximum window size (W in paper)
    int current_head;                   // Current write position (circular)
    int valid_length;                   // Number of valid entries
    int key_dim;                        // Key dimension
    int value_dim;                      // Value dimension
};

struct atlas_omega_rule {
    struct atlas_sliding_window window; // Sliding window buffer
    float alpha;                        // Learning rate for Omega updates
    float l2_regularization;            // L2 regularization coefficient
    int update_frequency;               // Update every N tokens
    bool adaptive_window;               // Dynamic window size adjustment
};
```

### 2. Core Implementation Functions
```c
// Omega Rule loss computation
struct ggml_tensor * atlas_omega_loss(
    struct ggml_context * ctx,
    struct atlas_omega_rule * omega,
    struct atlas_memory_module * memory,
    int current_position
);

// Sliding window management
int atlas_window_add_kv(
    struct atlas_sliding_window * window,
    const struct ggml_tensor * key,
    const struct ggml_tensor * value,
    int position
);

// Feature mapping application
struct ggml_tensor * atlas_apply_feature_mapping(
    struct ggml_context * ctx,
    const struct ggml_tensor * input,
    enum atlas_feature_type mapping_type
);
```

### 3. CUDA Optimization Requirements
```cuda
__global__ void atlas_omega_loss_kernel(
    const float* key_buffer,        // [window_size, key_dim]
    const float* value_buffer,      // [window_size, value_dim]  
    const bool* position_mask,      // [window_size] validity mask
    float* loss_output,             // [1] scalar loss
    int window_size,
    int current_position
);
```

## Testing Requirements

### Unit Tests
- [ ] **Sliding window operations**: Add, retrieve, circular buffer wraparound
- [ ] **Loss computation**: Verify L2 loss calculation against reference
- [ ] **Feature mapping**: Test polynomial and exponential mappings
- [ ] **Gradient computation**: Validate gradients for memory optimization
- [ ] **Boundary conditions**: Window start/end, empty windows, single elements

### Performance Tests
- [ ] **Sliding window efficiency**: O(1) add operations, O(w) loss computation
- [ ] **Memory bandwidth**: Optimize CUDA memory access patterns
- [ ] **Large window performance**: Test with windows up to 4K tokens

## Success Criteria
- [ ] Omega Rule loss computation matches reference implementation
- [ ] Sliding window operations work correctly with circular buffer
- [ ] CUDA implementation achieves >80% memory bandwidth utilization
- [ ] Integration with deep memory module from Issue #3
- [ ] All unit tests pass with 100% coverage

## Dependencies
- Issue #3: Deep Memory Module Implementation
- CUDA Toolkit 11.0+
- Feature mapping implementations

## Estimated Effort
**1.5 weeks** for experienced CUDA developer

## References
- ATLAS Paper Section 3.2: "Omega Rule"
- Sliding Window Attention research
