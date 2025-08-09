#include "atlas-gguf.h"
#include "common.h"

#include <cstdio>
#include <cstring>
#include <string>

// Simple ATLAS-GGUF conversion tool

void print_usage(const char * program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Convert between standard GGUF and ATLAS-enhanced GGUF formats\n\n");
    printf("Options:\n");
    printf("  -i, --input PATH        Input GGUF file path\n");
    printf("  -o, --output PATH       Output GGUF file path\n");
    printf("  --to-atlas              Convert standard GGUF to ATLAS-enhanced\n");
    printf("  --from-atlas            Convert ATLAS-enhanced GGUF to standard\n");
    printf("  --validate              Validate GGUF file only (no conversion)\n");
    printf("  --info                  Show file information only\n");
    printf("\n");
    printf("ATLAS Configuration (for --to-atlas):\n");
    printf("  --layer-count N         Number of layers (default: 32)\n");
    printf("  --head-count N          Number of attention heads (default: 32)\n");
    printf("  --batch-size N          Batch size (default: 4)\n");
    printf("  --seq-length N          Sequence length (default: 2048)\n");
    printf("  --storage-policy N      Storage policy: 0=none, 1=memory, 2=disk, 3=hybrid (default: 1)\n");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string input_path;
    std::string output_path;
    bool to_atlas = false;
    bool from_atlas = false;
    bool validate_only = false;
    bool show_info = false;
    
    // Default ATLAS configuration
    atlas_gguf_config_t atlas_config = {
        .version = 1,
        .enabled = true,
        .layer_count = 32,
        .head_count = 32,
        .batch_size = 4,
        .seq_length = 2048,
        .storage_policy = ATLAS_STORAGE_POLICY_MEMORY
    };

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" || arg == "--input") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            input_path = argv[++i];
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            output_path = argv[++i];
        }
        else if (arg == "--to-atlas") {
            to_atlas = true;
        }
        else if (arg == "--from-atlas") {
            from_atlas = true;
        }
        else if (arg == "--validate") {
            validate_only = true;
        }
        else if (arg == "--info") {
            show_info = true;
        }
        else if (arg == "--layer-count") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            atlas_config.layer_count = std::stoul(argv[++i]);
        }
        else if (arg == "--head-count") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            atlas_config.head_count = std::stoul(argv[++i]);
        }
        else if (arg == "--batch-size") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            atlas_config.batch_size = std::stoul(argv[++i]);
        }
        else if (arg == "--seq-length") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            atlas_config.seq_length = std::stoul(argv[++i]);
        }
        else if (arg == "--storage-policy") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
            int policy = std::stoi(argv[++i]);
            if (policy < 0 || policy > 3) {
                fprintf(stderr, "Error: Invalid storage policy: %d\n", policy);
                return 1;
            }
            atlas_config.storage_policy = (atlas_storage_policy_t)policy;
        }
        else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "Error: Unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    // Validate arguments
    if (input_path.empty()) {
        fprintf(stderr, "Error: Input path is required\n");
        return 1;
    }

    printf("ATLAS-GGUF Conversion Tool v%s\n", atlas_gguf_get_version_string());
    printf("==========================================\n");

    bool success = true;

    if (show_info) {
        printf("File Information: %s\n", input_path.c_str());
        struct gguf_init_params params = {false, NULL};
        atlas_gguf_context_t * ctx = atlas_gguf_init_from_file(input_path.c_str(), params);
        
        if (ctx) {
            atlas_gguf_print_info(ctx);
            atlas_gguf_free(ctx);
        } else {
            printf("Error: Failed to load GGUF file\n");
            success = false;
        }
    }
    else if (validate_only) {
        printf("Validating: %s\n", input_path.c_str());
        
        char error_msg[256];
        if (atlas_gguf_validate(input_path.c_str(), error_msg, sizeof(error_msg))) {
            printf("✓ Validation passed: %s\n", error_msg);
        } else {
            printf("✗ Validation failed: %s\n", error_msg);
            success = false;
        }
    }
    else if (to_atlas) {
        if (output_path.empty()) {
            fprintf(stderr, "Error: Output path is required for conversion\n");
            return 1;
        }
        
        printf("Converting to ATLAS format...\n");
        printf("  Input: %s\n", input_path.c_str());
        printf("  Output: %s\n", output_path.c_str());
        
        if (atlas_gguf_convert_to_atlas(input_path.c_str(), output_path.c_str(), &atlas_config)) {
            printf("✓ Conversion to ATLAS format completed successfully\n");
        } else {
            printf("✗ Conversion to ATLAS format failed\n");
            success = false;
        }
    }
    else if (from_atlas) {
        if (output_path.empty()) {
            fprintf(stderr, "Error: Output path is required for conversion\n");
            return 1;
        }
        
        printf("Converting from ATLAS format...\n");
        printf("  Input: %s\n", input_path.c_str());
        printf("  Output: %s\n", output_path.c_str());
        
        if (atlas_gguf_convert_from_atlas(input_path.c_str(), output_path.c_str())) {
            printf("✓ Conversion from ATLAS format completed successfully\n");
        } else {
            printf("✗ Conversion from ATLAS format failed\n");
            success = false;
        }
    }
    else {
        fprintf(stderr, "Error: Must specify an operation (--to-atlas, --from-atlas, --validate, or --info)\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("==========================================\n");
    if (success) {
        printf("Operation completed successfully\n");
        return 0;
    } else {
        printf("Operation failed\n");
        return 1;
    }
}