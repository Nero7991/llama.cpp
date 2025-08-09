#include "atlas-gguf.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>

#define ATLAS_GGUF_VERSION 1
#define ATLAS_GGUF_VERSION_STRING "1.0.0"

// Helper function to safely copy string
static void safe_strcpy(char * dst, const char * src, size_t dst_size) {
    if (dst_size > 0) {
        strncpy(dst, src, dst_size - 1);
        dst[dst_size - 1] = '\0';
    }
}

// Helper function to set error message
static void set_error(char * error_msg, size_t error_msg_size, const char * msg) {
    if (error_msg && error_msg_size > 0) {
        safe_strcpy(error_msg, msg, error_msg_size);
    }
}

atlas_gguf_context_t * atlas_gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    if (!fname) {
        return NULL;
    }

    atlas_gguf_context_t * ctx = (atlas_gguf_context_t*)calloc(1, sizeof(atlas_gguf_context_t));
    if (!ctx) {
        return NULL;
    }

    ctx->gguf_ctx = gguf_init_from_file(fname, params);
    if (!ctx->gguf_ctx) {
        free(ctx);
        return NULL;
    }

    // Load ATLAS configuration if present
    ctx->is_atlas_enabled = atlas_gguf_load_config(ctx->gguf_ctx, &ctx->config);

    return ctx;
}

atlas_gguf_context_t * atlas_gguf_init_empty(void) {
    atlas_gguf_context_t * ctx = (atlas_gguf_context_t*)calloc(1, sizeof(atlas_gguf_context_t));
    if (!ctx) {
        return NULL;
    }

    ctx->gguf_ctx = gguf_init_empty();
    if (!ctx->gguf_ctx) {
        free(ctx);
        return NULL;
    }

    // Initialize default ATLAS configuration
    ctx->config.version = ATLAS_GGUF_VERSION;
    ctx->config.enabled = false;
    ctx->config.layer_count = 0;
    ctx->config.head_count = 0;
    ctx->config.batch_size = 0;
    ctx->config.seq_length = 0;
    ctx->config.storage_policy = ATLAS_STORAGE_POLICY_NONE;
    ctx->is_atlas_enabled = false;

    return ctx;
}

void atlas_gguf_free(atlas_gguf_context_t * ctx) {
    if (!ctx) {
        return;
    }

    if (ctx->gguf_ctx) {
        gguf_free(ctx->gguf_ctx);
    }

    free(ctx);
}

bool atlas_gguf_has_atlas_data(const struct gguf_context * gguf_ctx) {
    if (!gguf_ctx) {
        return false;
    }

    // Check for ATLAS version key
    int key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_VERSION);
    if (key_id < 0) {
        return false;
    }

    // Check for ATLAS enabled key
    key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_ENABLED);
    return key_id >= 0;
}

bool atlas_gguf_load_config(const struct gguf_context * gguf_ctx, atlas_gguf_config_t * config) {
    if (!gguf_ctx || !config) {
        return false;
    }

    // Initialize with defaults
    memset(config, 0, sizeof(*config));
    config->version = ATLAS_GGUF_VERSION;
    config->storage_policy = ATLAS_STORAGE_POLICY_NONE;

    // Load version
    int key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_VERSION);
    if (key_id >= 0) {
        config->version = gguf_get_val_u32(gguf_ctx, key_id);
    }

    // Load enabled flag
    key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_ENABLED);
    if (key_id >= 0) {
        config->enabled = gguf_get_val_bool(gguf_ctx, key_id);
    } else {
        // If no enabled key, not an ATLAS file
        return false;
    }

    // Load other configuration parameters
    key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_LAYER_COUNT);
    if (key_id >= 0) {
        config->layer_count = gguf_get_val_u32(gguf_ctx, key_id);
    }

    key_id = gguf_find_key(gguf_ctx, ATLAS_GGUF_KEY_HEAD_COUNT);
    if (key_id >= 0) {
        config->head_count = gguf_get_val_u32(gguf_ctx, key_id);
    }

    return true;
}

bool atlas_gguf_save_config(struct gguf_context * gguf_ctx, const atlas_gguf_config_t * config) {
    if (!gguf_ctx || !config) {
        return false;
    }

    // Save version
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_VERSION, config->version);

    // Save enabled flag
    gguf_set_val_bool(gguf_ctx, ATLAS_GGUF_KEY_ENABLED, config->enabled);

    // Save configuration parameters
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_LAYER_COUNT, config->layer_count);
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_HEAD_COUNT, config->head_count);
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_BATCH_SIZE, config->batch_size);
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_SEQ_LENGTH, config->seq_length);
    gguf_set_val_u32(gguf_ctx, ATLAS_GGUF_KEY_STORAGE_POLICY, (uint32_t)config->storage_policy);

    return true;
}

int atlas_gguf_get_atlas_tensor_count(const struct gguf_context * gguf_ctx) {
    if (!gguf_ctx) {
        return 0;
    }

    int total_tensors = gguf_get_n_tensors(gguf_ctx);
    int atlas_count = 0;

    for (int i = 0; i < total_tensors; i++) {
        const char * tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        if (tensor_name && atlas_gguf_is_atlas_tensor(tensor_name)) {
            atlas_count++;
        }
    }

    return atlas_count;
}

bool atlas_gguf_is_atlas_tensor(const char * name) {
    if (!name) {
        return false;
    }

    return strncmp(name, ATLAS_TENSOR_PREFIX, strlen(ATLAS_TENSOR_PREFIX)) == 0;
}

void atlas_gguf_make_tensor_name(char * buffer, size_t buffer_size, const char * prefix, 
                                 int layer_idx, const char * component) {
    if (!buffer || buffer_size == 0) {
        return;
    }

    if (!prefix) {
        prefix = ATLAS_TENSOR_PREFIX;
    }

    if (layer_idx >= 0 && component) {
        snprintf(buffer, buffer_size, "%s%d.%s", prefix, layer_idx, component);
    } else if (component) {
        snprintf(buffer, buffer_size, "%s%s", prefix, component);
    } else {
        safe_strcpy(buffer, prefix, buffer_size);
    }
}

bool atlas_gguf_convert_to_atlas(const char * input_path, const char * output_path, 
                                 const atlas_gguf_config_t * config) {
    if (!input_path || !output_path || !config) {
        return false;
    }

    // Load input GGUF file
    struct gguf_init_params params = {false, NULL};
    struct gguf_context * input_ctx = gguf_init_from_file(input_path, params);
    if (!input_ctx) {
        return false;
    }

    // Create new GGUF context for output
    struct gguf_context * output_ctx = gguf_init_empty();
    if (!output_ctx) {
        gguf_free(input_ctx);
        return false;
    }

    // Copy all existing key-value pairs
    gguf_set_kv(output_ctx, input_ctx);

    // Add ATLAS configuration
    bool success = atlas_gguf_save_config(output_ctx, config);
    
    if (success) {
        // Write output file
        gguf_write_to_file(output_ctx, output_path, false);
    }

    gguf_free(output_ctx);
    gguf_free(input_ctx);

    return success;
}

bool atlas_gguf_convert_from_atlas(const char * input_path, const char * output_path) {
    if (!input_path || !output_path) {
        return false;
    }

    // Load ATLAS-enhanced GGUF file
    struct gguf_init_params params = {false, NULL};
    atlas_gguf_context_t * atlas_ctx = atlas_gguf_init_from_file(input_path, params);
    if (!atlas_ctx) {
        return false;
    }

    if (!atlas_ctx->is_atlas_enabled) {
        // Already a standard GGUF file
        atlas_gguf_free(atlas_ctx);
        return false;
    }

    // Create new standard GGUF context
    struct gguf_context * output_ctx = gguf_init_empty();
    if (!output_ctx) {
        atlas_gguf_free(atlas_ctx);
        return false;
    }

    // Copy non-ATLAS key-value pairs
    int n_kv = gguf_get_n_kv(atlas_ctx->gguf_ctx);
    for (int i = 0; i < n_kv; i++) {
        const char * key = gguf_get_key(atlas_ctx->gguf_ctx, i);
        if (key && strncmp(key, ATLAS_GGUF_KEY_PREFIX, strlen(ATLAS_GGUF_KEY_PREFIX)) != 0) {
            // Copy non-ATLAS keys (simplified approach)
            enum gguf_type type = gguf_get_kv_type(atlas_ctx->gguf_ctx, i);
            switch (type) {
                case GGUF_TYPE_BOOL:
                    gguf_set_val_bool(output_ctx, key, gguf_get_val_bool(atlas_ctx->gguf_ctx, i));
                    break;
                case GGUF_TYPE_UINT32:
                    gguf_set_val_u32(output_ctx, key, gguf_get_val_u32(atlas_ctx->gguf_ctx, i));
                    break;
                case GGUF_TYPE_STRING:
                    gguf_set_val_str(output_ctx, key, gguf_get_val_str(atlas_ctx->gguf_ctx, i));
                    break;
                default:
                    break;
            }
        }
    }

    // Write output file
    gguf_write_to_file(output_ctx, output_path, false);

    gguf_free(output_ctx);
    atlas_gguf_free(atlas_ctx);

    return true;
}

bool atlas_gguf_validate(const char * path, char * error_msg, size_t error_msg_size) {
    if (!path) {
        set_error(error_msg, error_msg_size, "Invalid file path");
        return false;
    }

    struct gguf_init_params params = {false, NULL};
    atlas_gguf_context_t * ctx = atlas_gguf_init_from_file(path, params);
    if (!ctx) {
        set_error(error_msg, error_msg_size, "Failed to load GGUF file");
        return false;
    }

    bool is_valid = true;

    if (ctx->is_atlas_enabled) {
        // Validate ATLAS configuration
        if (ctx->config.version == 0) {
            set_error(error_msg, error_msg_size, "Invalid ATLAS version");
            is_valid = false;
        }
    }

    atlas_gguf_free(ctx);

    if (is_valid && error_msg) {
        set_error(error_msg, error_msg_size, "File validation passed");
    }

    return is_valid;
}

const char * atlas_gguf_get_version_string(void) {
    return ATLAS_GGUF_VERSION_STRING;
}

void atlas_gguf_print_info(const atlas_gguf_context_t * ctx) {
    if (!ctx) {
        printf("Invalid ATLAS-GGUF context\n");
        return;
    }

    printf("ATLAS-GGUF Information:\n");
    printf("  Version: %u (%s)\n", ctx->config.version, atlas_gguf_get_version_string());
    printf("  ATLAS Enabled: %s\n", ctx->is_atlas_enabled ? "Yes" : "No");

    if (ctx->is_atlas_enabled) {
        printf("  Configuration:\n");
        printf("    Layer Count: %u\n", ctx->config.layer_count);
        printf("    Head Count: %u\n", ctx->config.head_count);
        printf("    Batch Size: %u\n", ctx->config.batch_size);
        printf("    Sequence Length: %u\n", ctx->config.seq_length);
        printf("    Storage Policy: %u\n", ctx->config.storage_policy);

        int atlas_tensors = atlas_gguf_get_atlas_tensor_count(ctx->gguf_ctx);
        printf("    ATLAS Tensors: %d\n", atlas_tensors);
    }

    int total_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    int total_kv = gguf_get_n_kv(ctx->gguf_ctx);
    printf("  Total Tensors: %d\n", total_tensors);
    printf("  Total Key-Value Pairs: %d\n", total_kv);
}