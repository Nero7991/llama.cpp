#pragma once

#include "ggml.h"
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

// ATLAS metadata keys for GGUF format
#define ATLAS_GGUF_KEY_PREFIX "atlas."

// Core ATLAS metadata keys
#define ATLAS_GGUF_KEY_VERSION          "atlas.version"
#define ATLAS_GGUF_KEY_ENABLED          "atlas.enabled"
#define ATLAS_GGUF_KEY_CONFIG           "atlas.config"
#define ATLAS_GGUF_KEY_STORAGE_VERSION  "atlas.storage_version"

// ATLAS configuration keys
#define ATLAS_GGUF_KEY_LAYER_COUNT      "atlas.layer_count"
#define ATLAS_GGUF_KEY_HEAD_COUNT       "atlas.head_count"
#define ATLAS_GGUF_KEY_BATCH_SIZE       "atlas.batch_size"
#define ATLAS_GGUF_KEY_SEQ_LENGTH       "atlas.seq_length"
#define ATLAS_GGUF_KEY_STORAGE_POLICY   "atlas.storage_policy"

// ATLAS tensor naming conventions
#define ATLAS_TENSOR_PREFIX             "atlas."
#define ATLAS_TENSOR_ATTENTION_PREFIX   "atlas.attn."
#define ATLAS_TENSOR_FFN_PREFIX         "atlas.ffn."
#define ATLAS_TENSOR_STATE_PREFIX       "atlas.state."

// ATLAS storage policies
typedef enum {
    ATLAS_STORAGE_POLICY_NONE = 0,
    ATLAS_STORAGE_POLICY_MEMORY,
    ATLAS_STORAGE_POLICY_DISK,
    ATLAS_STORAGE_POLICY_HYBRID,
} atlas_storage_policy_t;

// ATLAS configuration structure
typedef struct {
    uint32_t version;
    bool enabled;
    uint32_t layer_count;
    uint32_t head_count;
    uint32_t batch_size;
    uint32_t seq_length;
    atlas_storage_policy_t storage_policy;
} atlas_gguf_config_t;

// ATLAS context for GGUF operations
typedef struct {
    struct gguf_context * gguf_ctx;
    atlas_gguf_config_t config;
    bool is_atlas_enabled;
} atlas_gguf_context_t;

// Core ATLAS-GGUF API functions

// Initialize ATLAS-GGUF context from file
atlas_gguf_context_t * atlas_gguf_init_from_file(const char * fname, struct gguf_init_params params);

// Initialize empty ATLAS-GGUF context
atlas_gguf_context_t * atlas_gguf_init_empty(void);

// Free ATLAS-GGUF context
void atlas_gguf_free(atlas_gguf_context_t * ctx);

// Check if GGUF file has ATLAS metadata
bool atlas_gguf_has_atlas_data(const struct gguf_context * gguf_ctx);

// Load ATLAS configuration from GGUF context
bool atlas_gguf_load_config(const struct gguf_context * gguf_ctx, atlas_gguf_config_t * config);

// Save ATLAS configuration to GGUF context
bool atlas_gguf_save_config(struct gguf_context * gguf_ctx, const atlas_gguf_config_t * config);

// Get ATLAS tensor count in GGUF file
int atlas_gguf_get_atlas_tensor_count(const struct gguf_context * gguf_ctx);

// Check if tensor name follows ATLAS naming convention
bool atlas_gguf_is_atlas_tensor(const char * name);

// Generate ATLAS tensor name
void atlas_gguf_make_tensor_name(char * buffer, size_t buffer_size, const char * prefix, 
                                 int layer_idx, const char * component);

// Conversion utilities

// Convert standard GGUF to ATLAS-enhanced GGUF
bool atlas_gguf_convert_to_atlas(const char * input_path, const char * output_path, 
                                 const atlas_gguf_config_t * config);

// Convert ATLAS-enhanced GGUF to standard GGUF
bool atlas_gguf_convert_from_atlas(const char * input_path, const char * output_path);

// Validate ATLAS-GGUF file integrity
bool atlas_gguf_validate(const char * path, char * error_msg, size_t error_msg_size);

// Utility functions

// Get ATLAS version string
const char * atlas_gguf_get_version_string(void);

// Print ATLAS metadata information
void atlas_gguf_print_info(const atlas_gguf_context_t * ctx);

#ifdef __cplusplus
}
#endif