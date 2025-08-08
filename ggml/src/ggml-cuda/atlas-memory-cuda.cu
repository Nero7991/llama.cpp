#include "ggml-atlas-memory.h"
#include <math.h>

#ifdef GGML_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define ATLAS_CUDA_BLOCK_SIZE 256
#define ATLAS_CUDA_MAX_THREADS_PER_BLOCK 1024

// CUDA kernels for activation functions
__device__ float atlas_cuda_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

__device__ float atlas_cuda_relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float atlas_cuda_silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void atlas_cuda_activation_kernel(
    float * data,
    int size,
    int activation_type) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float val = data[idx];
    switch (activation_type) {
        case 0: // GELU
            data[idx] = atlas_cuda_gelu(val);
            break;
        case 1: // ReLU
            data[idx] = atlas_cuda_relu(val);
            break;
        case 2: // SiLU
            data[idx] = atlas_cuda_silu(val);
            break;
    }
}

// Optimized 2-layer MLP kernel with shared memory
__global__ void atlas_cuda_mlp_kernel(
    const float * input,        // [batch_size, input_dim]
    const float * w1,          // [input_dim, hidden_dim]
    const float * b1,          // [hidden_dim]
    const float * w2,          // [hidden_dim, output_dim]
    const float * b2,          // [output_dim]
    float * output,            // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int hidden_dim,
    int output_dim,
    int activation_type,
    bool use_residual) {
    
    extern __shared__ float shared_mem[];
    
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Shared memory layout: [hidden_output, temp_storage]
    float * hidden_output = shared_mem;
    float * temp_storage = shared_mem + hidden_dim;
    
    const float * input_row = input + batch_idx * input_dim;
    float * output_row = output + batch_idx * output_dim;
    
    // First layer: input -> hidden
    for (int h = thread_idx; h < hidden_dim; h += blockDim.x) {
        float sum = b1[h];
        for (int i = 0; i < input_dim; i++) {
            sum += input_row[i] * w1[i * hidden_dim + h];
        }
        hidden_output[h] = sum;
    }
    
    __syncthreads();
    
    // Apply activation
    for (int h = thread_idx; h < hidden_dim; h += blockDim.x) {
        float val = hidden_output[h];
        switch (activation_type) {
            case 0: // GELU
                hidden_output[h] = atlas_cuda_gelu(val);
                break;
            case 1: // ReLU
                hidden_output[h] = atlas_cuda_relu(val);
                break;
            case 2: // SiLU
                hidden_output[h] = atlas_cuda_silu(val);
                break;
        }
    }
    
    __syncthreads();
    
    // Second layer: hidden -> output
    for (int o = thread_idx; o < output_dim; o += blockDim.x) {
        float sum = b2[o];
        for (int h = 0; h < hidden_dim; h++) {
            sum += hidden_output[h] * w2[h * output_dim + o];
        }
        
        // Apply residual connection if enabled
        if (use_residual && input_dim == output_dim) {
            sum += input_row[o];
        }
        
        output_row[o] = sum;
    }
}

bool ggml_atlas_memory_cuda_supported(void) {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

struct ggml_tensor * ggml_atlas_memory_cuda_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input) {
    
    const struct ggml_atlas_memory_config * config = &atlas_ctx->config;
    
    // Validate input dimensions
    if (input->ne[0] != config->input_dim) {
        return NULL;
    }
    
    int batch_size = input->ne[1];
    
    // Create output tensor
    struct ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config->output_dim, batch_size);
    
    // Get CUDA pointers
    float * d_input = (float *)input->data;
    float * d_w1 = (float *)atlas_ctx->w1->data;
    float * d_b1 = (float *)atlas_ctx->b1->data;
    float * d_w2 = (float *)atlas_ctx->w2->data;
    float * d_b2 = (float *)atlas_ctx->b2->data;
    float * d_output = (float *)output->data;
    
    // Calculate shared memory size
    size_t shared_mem_size = (config->hidden_dim + config->hidden_dim) * sizeof(float);
    
    // Launch optimized MLP kernel
    dim3 block_dim(min(ATLAS_CUDA_MAX_THREADS_PER_BLOCK, 
                      max(config->hidden_dim, config->output_dim)));
    dim3 grid_dim(batch_size);
    
    atlas_cuda_mlp_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        d_input,
        d_w1,
        d_b1,
        d_w2,
        d_b2,
        d_output,
        batch_size,
        config->input_dim,
        config->hidden_dim,
        config->output_dim,
        (int)config->activation,
        config->use_residual
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return NULL;
    }
    
    return output;
}

#else // !GGML_USE_CUDA

bool ggml_atlas_memory_cuda_supported(void) {
    return false;
}

struct ggml_tensor * ggml_atlas_memory_cuda_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input) {
    return NULL;
}

#endif // GGML_USE_CUDA