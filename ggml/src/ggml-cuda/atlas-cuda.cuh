#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../../include/atlas/atlas-types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS CUDA kernel launch parameters
struct atlas_cuda_kernel_params {
    int batch_size;
    int seq_len;
    int hidden_dim;
    int memory_depth;
    float learning_rate;
    float momentum;
    int newton_schulz_iterations;
    bool use_tensor_cores;
    cudaStream_t stream;
};

// ATLAS CUDA memory layout
struct atlas_cuda_memory_layout {
    // Input/Output tensors
    float* input;           // [batch_size, seq_len, hidden_dim]
    float* output;          // [batch_size, seq_len, hidden_dim] 
    
    // Deep Memory Module weights
    __half* w1;             // [hidden_dim, memory_depth]
    __half* b1;             // [memory_depth]
    __half* w2;             // [memory_depth, hidden_dim]
    __half* b2;             // [hidden_dim]
    __half* w_res;          // [hidden_dim, hidden_dim]
    
    // Omega Rule state
    float* omega_keys;      // [window_size, hidden_dim]
    float* omega_values;    // [window_size, hidden_dim]
    float* omega_weights;   // [window_size]
    
    // Muon Optimizer state
    float* momentum_state;  // [hidden_dim, hidden_dim]
    float* newton_schulz_workspace; // Temporary workspace
    
    // Feature mapping workspace  
    float* poly_features;   // Polynomial feature expansion
    float* fourier_workspace; // FFT workspace if needed
    
    // Shared workspace
    float* temp_workspace;  // General temporary storage
    size_t workspace_size;
};

// Kernel function declarations

// Fused ATLAS forward pass kernel
__global__ void atlas_fused_forward_kernel(
    const float* __restrict__ input,
    const __half* __restrict__ weights_packed,
    float* __restrict__ output,
    float* __restrict__ workspace,
    atlas_cuda_kernel_params params
);

// Feature mapping kernels
__global__ void atlas_polynomial_features_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int degree,
    int batch_size,
    int seq_len,
    int hidden_dim
);

__global__ void atlas_fourier_features_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int num_freqs,
    int batch_size,
    int seq_len,
    int hidden_dim
);

// Deep memory module kernels  
__global__ void atlas_deep_memory_forward_kernel(
    const float* __restrict__ input,
    const __half* __restrict__ w1,
    const __half* __restrict__ b1, 
    const __half* __restrict__ w2,
    const __half* __restrict__ b2,
    const __half* __restrict__ w_res,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int memory_depth
);

// Omega Rule kernels
__global__ void atlas_omega_update_kernel(
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ weights,
    float* __restrict__ omega_keys,
    float* __restrict__ omega_values,
    float learning_rate,
    int window_size,
    int hidden_dim,
    int current_pos
);

__global__ void atlas_sliding_window_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ omega_keys,
    const float* __restrict__ omega_values,
    const float* __restrict__ omega_weights,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int window_size
);

// Muon Optimizer kernels  
__global__ void atlas_newton_schulz_kernel(
    float* __restrict__ matrix,
    float* __restrict__ inverse,
    float* __restrict__ workspace,
    int dim,
    int iterations
);

__global__ void atlas_muon_momentum_update_kernel(
    const float* __restrict__ gradients,
    float* __restrict__ momentum,
    float* __restrict__ params,
    float learning_rate,
    float beta,
    int size
);

// Utility kernels
__global__ void atlas_activation_kernel(
    float* __restrict__ data,
    int activation_type,
    int size
);

__global__ void atlas_layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float epsilon,
    int batch_size,
    int seq_len,
    int hidden_dim
);

// Memory management functions
cudaError_t atlas_cuda_malloc_layout(
    atlas_cuda_memory_layout* layout,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int memory_depth,
    int window_size
);

void atlas_cuda_free_layout(atlas_cuda_memory_layout* layout);

// Performance optimization functions
void atlas_cuda_select_optimal_kernels(
    atlas_cuda_kernel_params* params,
    int device_id
);

// Multi-GPU support
cudaError_t atlas_cuda_multi_gpu_setup(
    int num_devices,
    int* device_ids,
    atlas_cuda_memory_layout* layouts
);

cudaError_t atlas_cuda_all_reduce(
    float* data,
    int count,
    int num_devices,
    cudaStream_t* streams
);

// Tensor Core optimizations (when available)
#if __CUDA_ARCH__ >= 700
__global__ void atlas_tensor_core_gemm_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
);
#endif

// Performance monitoring
struct atlas_cuda_perf_counters {
    float kernel_time_ms;
    float memory_bandwidth_gbps; 
    float compute_utilization;
    float tensor_core_utilization;
    size_t memory_used_bytes;
    size_t peak_memory_bytes;
};

void atlas_cuda_get_perf_counters(
    atlas_cuda_perf_counters* counters,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif