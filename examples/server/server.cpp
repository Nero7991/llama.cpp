#include "../../include/atlas-persistence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

// External C API functions from atlas-persistence-server.cpp
extern "C" {
    bool atlas_initialize(const char* model_path, const char* atlas_file);
    void atlas_shutdown(void);
    bool atlas_store_memory(const char* key, const char* content, const char* type);
    bool atlas_retrieve_memory(const char* key, char** content, char** type);
    bool atlas_has_memory(const char* key);
    bool atlas_remove_memory(const char* key);
    size_t atlas_memory_count(void);
    bool atlas_save_memory(void);
    void atlas_set_auto_save(bool enabled);
    const char* atlas_get_filepath(void);
    void atlas_register_signal_handlers(void);
    void run_atlas_demo(const char* model_path);
}

// Simple server demo for ATLAS memory persistence
// This demonstrates the basic functionality for Issue #12

struct ServerParams {
    char model_path[512];
    char atlas_file[512]; 
    bool atlas_enabled;
    bool atlas_no_auto_save;
    bool verbose;
};

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("\nOptions:\n");
    printf("  -m, --model PATH            Model file path (.gguf)\n");
    printf("  --atlas                     Enable ATLAS memory persistence\n");
    printf("  --atlas-file PATH           Custom ATLAS memory file path\n");
    printf("  --atlas-no-auto-save        Disable automatic memory saving\n");
    printf("  -v, --verbose               Enable verbose logging\n");
    printf("  -h, --help                  Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -m model.gguf --atlas\n", program_name);
    printf("  %s -m model.gguf --atlas --atlas-file custom.atlas\n", program_name);
    printf("  echo 'store test_key test_value' | %s -m model.gguf --atlas\n", program_name);
}

bool parse_args(int argc, char** argv, ServerParams* params) {
    // Initialize defaults
    memset(params, 0, sizeof(ServerParams));
    params->atlas_enabled = false;
    params->atlas_no_auto_save = false;
    params->verbose = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --model requires a value\n");
                return false;
            }
            strncpy(params->model_path, argv[++i], sizeof(params->model_path) - 1);
        }
        else if (strcmp(argv[i], "--atlas") == 0) {
            params->atlas_enabled = true;
        }
        else if (strcmp(argv[i], "--atlas-file") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --atlas-file requires a value\n");
                return false;
            }
            strncpy(params->atlas_file, argv[++i], sizeof(params->atlas_file) - 1);
        }
        else if (strcmp(argv[i], "--atlas-no-auto-save") == 0) {
            params->atlas_no_auto_save = true;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            params->verbose = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return false;
        }
    }
    
    // Validate required parameters
    if (strlen(params->model_path) == 0) {
        fprintf(stderr, "Error: Model path is required (use -m or --model)\n");
        return false;
    }
    
    return true;
}

// Interactive CLI for demonstration
void run_interactive_cli(const ServerParams* params) {
    printf("\n=== ATLAS Memory Persistence Demo ===\n");
    printf("Model: %s\n", params->model_path);
    
    if (params->atlas_enabled) {
        const char* atlas_file = strlen(params->atlas_file) > 0 ? 
                                params->atlas_file : nullptr;
        
        if (!atlas_initialize(params->model_path, atlas_file)) {
            printf("Failed to initialize ATLAS persistence\n");
            return;
        }
        
        if (!params->atlas_no_auto_save) {
            atlas_set_auto_save(true);
            printf("Auto-save: enabled\n");
        } else {
            atlas_set_auto_save(false);
            printf("Auto-save: disabled\n");
        }
        
        printf("Atlas file: %s\n", atlas_get_filepath());
        printf("Current entries: %zu\n", atlas_memory_count());
        
        // Register signal handlers for graceful shutdown
        atlas_register_signal_handlers();
        
        // Run the interactive demo
        run_atlas_demo(params->model_path);
    } else {
        printf("ATLAS not enabled. Use --atlas to enable memory persistence.\n");
    }
}

// Batch processing from stdin
void run_batch_processing(const ServerParams* params) {
    if (!params->atlas_enabled) {
        printf("ATLAS not enabled for batch processing\n");
        return;
    }
    
    // Initialize ATLAS
    const char* atlas_file = strlen(params->atlas_file) > 0 ? 
                            params->atlas_file : nullptr;
    if (!atlas_initialize(params->model_path, atlas_file)) {
        printf("Failed to initialize ATLAS persistence\n");
        return;
    }
    
    if (!params->atlas_no_auto_save) {
        atlas_set_auto_save(true);
    }
    
    if (params->verbose) {
        printf("ATLAS initialized: %s\n", atlas_get_filepath());
        printf("Processing stdin commands...\n");
    }
    
    // Process commands from stdin
    char line[1024];
    char command[64], key[256], content[512], type[32];
    int line_num = 0;
    
    while (fgets(line, sizeof(line), stdin)) {
        line_num++;
        line[strcspn(line, "\n")] = '\0';  // Remove newline
        
        if (strlen(line) == 0) continue;
        
        int args = sscanf(line, "%63s %255s %511s %31s", command, key, content, type);
        
        if (strcmp(command, "store") == 0 && args >= 3) {
            const char* entry_type = (args >= 4) ? type : "text";
            if (atlas_store_memory(key, content, entry_type)) {
                if (params->verbose) {
                    printf("Line %d: Stored %s -> %s (%s)\n", 
                           line_num, key, content, entry_type);
                }
            } else {
                fprintf(stderr, "Line %d: Failed to store %s\n", line_num, key);
            }
        }
        else if (strcmp(command, "get") == 0 && args >= 2) {
            char* retrieved_content = nullptr;
            char* retrieved_type = nullptr;
            if (atlas_retrieve_memory(key, &retrieved_content, &retrieved_type)) {
                printf("%s -> %s (%s)\n", key, retrieved_content, 
                       retrieved_type ? retrieved_type : "unknown");
                free(retrieved_content);
                free(retrieved_type);
            } else {
                if (params->verbose) {
                    fprintf(stderr, "Line %d: Key not found: %s\n", line_num, key);
                }
            }
        }
        else if (strcmp(command, "remove") == 0 && args >= 2) {
            if (atlas_remove_memory(key)) {
                if (params->verbose) {
                    printf("Line %d: Removed %s\n", line_num, key);
                }
            } else {
                fprintf(stderr, "Line %d: Failed to remove %s\n", line_num, key);
            }
        }
        else if (strcmp(command, "count") == 0) {
            printf("%zu\n", atlas_memory_count());
        }
        else if (strcmp(command, "save") == 0) {
            if (atlas_save_memory()) {
                if (params->verbose) {
                    printf("Line %d: Memory saved\n", line_num);
                }
            } else {
                fprintf(stderr, "Line %d: Failed to save memory\n", line_num);
            }
        }
        else {
            if (params->verbose) {
                fprintf(stderr, "Line %d: Unknown command: %s\n", line_num, command);
            }
        }
    }
    
    if (params->verbose) {
        printf("Batch processing complete. Final count: %zu entries\n", 
               atlas_memory_count());
    }
    
    atlas_shutdown();
}

int main(int argc, char** argv) {
    ServerParams params;
    
    if (!parse_args(argc, argv, &params)) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Check if we're running interactively (stdin is a terminal)
    if (isatty(STDIN_FILENO)) {
        run_interactive_cli(&params);
    } else {
        run_batch_processing(&params);
    }
    
    return 0;
}