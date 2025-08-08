#include "ggml-atlas-memory.h"
#include "ggml-impl.h"
#include <math.h>
#include <string.h>

// Activation functions
static inline float atlas_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

static inline float atlas_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float atlas_silu(float x) {
    return x / (1.0f + expf(-x));
}

bool ggml_atlas_memory_cpu_supported(void) {
    return true;
}

struct ggml_tensor * ggml_atlas_memory_cpu_forward(
    struct ggml_context * ctx,
    struct ggml_atlas_memory_context * atlas_ctx,
    struct ggml_tensor * input) {
    
    const struct ggml_atlas_memory_config * config = &atlas_ctx->config;
    
    // Validate input dimensions
    if (input->ne[0] != config->input_dim) {
        return NULL;
    }
    
    int batch_size = input->ne[1];
    
    // First layer: input -> hidden
    struct ggml_tensor * hidden = ggml_mul_mat(ctx, atlas_ctx->w1, input);
    hidden = ggml_add(ctx, hidden, ggml_repeat(ctx, atlas_ctx->b1, hidden));
    
    // Apply activation
    struct ggml_tensor * activated;
    switch (config->activation) {
        case GGML_ATLAS_ACT_GELU:
            activated = ggml_gelu(ctx, hidden);
            break;
        case GGML_ATLAS_ACT_RELU:
            activated = ggml_relu(ctx, hidden);
            break;
        case GGML_ATLAS_ACT_SILU:
            activated = ggml_silu(ctx, hidden);
            break;
        default:
            activated = hidden;
    }
    
    // Second layer: hidden -> output
    struct ggml_tensor * output = ggml_mul_mat(ctx, atlas_ctx->w2, activated);
    output = ggml_add(ctx, output, ggml_repeat(ctx, atlas_ctx->b2, output));
    
    // Residual connection if enabled and dimensions match
    if (config->use_residual && config->input_dim == config->output_dim) {
        output = ggml_add(ctx, output, input);
    }
    
    return output;
}