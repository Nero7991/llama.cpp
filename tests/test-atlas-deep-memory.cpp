#include "ggml.h"
#include "ggml-atlas-memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

#define TEST_BATCH_SIZE 2
#define TEST_INPUT_DIM 64
#define TEST_HIDDEN_DIM 128
#define TEST_OUTPUT_DIM 32

// Test configuration
static struct ggml_atlas_memory_config test_config = {
    .input_dim = TEST_INPUT_DIM,
    .hidden_dim = TEST_HIDDEN_DIM,
    .output_dim = TEST_OUTPUT_DIM,
    .activation = GGML_ATLAS_ACT_GELU,
    .dropout_rate = 0.0f,
    .use_residual = false,
};

static bool test_atlas_memory_init_free(void) {
    printf("Testing ATLAS memory initialization and cleanup...\n");
    
    struct ggml_atlas_memory_context * ctx = ggml_atlas_memory_init(&test_config);
    if (!ctx) {
        printf("FAILED: Could not initialize ATLAS memory context\n");
        return false;
    }
    
    // Check that weights are initialized
    if (!ctx->w1 || !ctx->b1 || !ctx->w2 || !ctx->b2) {
        printf("FAILED: Weights not properly initialized\n");
        ggml_atlas_memory_free(ctx);
        return false;
    }
    
    // Check dimensions
    if (ctx->w1->ne[0] != TEST_INPUT_DIM || ctx->w1->ne[1] != TEST_HIDDEN_DIM) {
        printf("FAILED: W1 dimensions incorrect\n");
        ggml_atlas_memory_free(ctx);
        return false;
    }
    
    if (ctx->w2->ne[0] != TEST_HIDDEN_DIM || ctx->w2->ne[1] != TEST_OUTPUT_DIM) {
        printf("FAILED: W2 dimensions incorrect\n");
        ggml_atlas_memory_free(ctx);
        return false;
    }
    
    ggml_atlas_memory_free(ctx);
    printf("PASSED: Initialization and cleanup test\n");
    return true;
}

int main(void) {
    printf("Running ATLAS Phase 2A Deep Memory Module Tests\n");
    printf("===============================================\n");
    
    srand(42); // Reproducible random numbers
    
    bool all_passed = true;
    
    if (!ggml_atlas_memory_supported()) {
        printf("ATLAS memory module not supported on this system\n");
        return 1;
    }
    
    all_passed &= test_atlas_memory_init_free();
    
    printf("\n===============================================\n");
    if (all_passed) {
        printf("All tests PASSED\n");
        return 0;
    } else {
        printf("Some tests FAILED\n");
        return 1;
    }
}