#include "../../include/atlas-persistence.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>

// Global persistence context
static atlas_persistence_t* g_atlas_ctx = nullptr;
static char g_atlas_filepath[512] = {0};
static bool g_auto_save_enabled = true;
static bool g_shutdown_requested = false;
static pthread_mutex_t g_atlas_mutex = PTHREAD_MUTEX_INITIALIZER;

// C-style API for server integration
extern "C" {

// Initialize ATLAS persistence
bool atlas_initialize(const char* model_path, const char* atlas_file) {
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return true;  // Already initialized
    }
    
    g_atlas_ctx = atlas_persistence_create();
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    // Determine atlas file path
    if (atlas_file && strlen(atlas_file) > 0) {
        strncpy(g_atlas_filepath, atlas_file, sizeof(g_atlas_filepath) - 1);
        g_atlas_filepath[sizeof(g_atlas_filepath) - 1] = '\0';
    } else {
        // Generate automatic filename
        char auto_filename[512];
        atlas_get_auto_filename(model_path, auto_filename);
        strncpy(g_atlas_filepath, auto_filename, sizeof(g_atlas_filepath) - 1);
        g_atlas_filepath[sizeof(g_atlas_filepath) - 1] = '\0';
    }
    
    // Load existing file if it exists
    if (atlas_persistence_exists(g_atlas_filepath)) {
        if (atlas_persistence_load(g_atlas_ctx, g_atlas_filepath)) {
            printf("ATLAS: Loaded existing memory from %s (%zu entries)\n", 
                   g_atlas_filepath, atlas_persistence_count(g_atlas_ctx));
        } else {
            printf("ATLAS: Failed to load %s: %s\n", 
                   g_atlas_filepath, atlas_error_string(atlas_get_last_error()));
        }
    } else {
        printf("ATLAS: Starting with fresh memory, will save to %s\n", g_atlas_filepath);
    }
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return true;
}

// Shutdown and cleanup
void atlas_shutdown(void) {
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (g_atlas_ctx && g_auto_save_enabled && !g_shutdown_requested) {
        printf("ATLAS: Saving memory state to %s...\n", g_atlas_filepath);
        if (atlas_persistence_save(g_atlas_ctx, g_atlas_filepath)) {
            printf("ATLAS: Memory saved successfully (%zu entries)\n", 
                   atlas_persistence_count(g_atlas_ctx));
        } else {
            printf("ATLAS: Failed to save memory: %s\n", 
                   atlas_error_string(atlas_get_last_error()));
        }
    }
    
    if (g_atlas_ctx) {
        atlas_persistence_free(g_atlas_ctx);
        g_atlas_ctx = nullptr;
    }
    
    pthread_mutex_unlock(&g_atlas_mutex);
}

// Store memory entry
bool atlas_store_memory(const char* key, const char* content, const char* type) {
    if (!key || !content) return false;
    
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    uint32_t type_id = 0;  // Default type
    if (type) {
        if (strcmp(type, "text") == 0) type_id = 1;
        else if (strcmp(type, "conversation") == 0) type_id = 2;
        else if (strcmp(type, "context") == 0) type_id = 3;
        else if (strcmp(type, "metadata") == 0) type_id = 4;
    }
    
    bool result = atlas_persistence_set(g_atlas_ctx, key, content, 
                                       strlen(content), type_id);
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return result;
}

// Retrieve memory entry
bool atlas_retrieve_memory(const char* key, char** content, char** type) {
    if (!key || !content) return false;
    
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    void* data;
    size_t data_size;
    uint32_t type_id;
    
    bool result = atlas_persistence_get(g_atlas_ctx, key, &data, &data_size, &type_id);
    if (result && data) {
        // Allocate null-terminated string for content
        *content = (char*)malloc(data_size + 1);
        if (*content) {
            memcpy(*content, data, data_size);
            (*content)[data_size] = '\0';
            
            // Set type string if requested
            if (type) {
                switch (type_id) {
                    case 1: *type = strdup("text"); break;
                    case 2: *type = strdup("conversation"); break;
                    case 3: *type = strdup("context"); break;
                    case 4: *type = strdup("metadata"); break;
                    default: *type = strdup("unknown"); break;
                }
            }
        } else {
            result = false;
        }
        free(data);
    }
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return result;
}

// Check if memory entry exists
bool atlas_has_memory(const char* key) {
    if (!key) return false;
    
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    bool result = atlas_persistence_has(g_atlas_ctx, key);
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return result;
}

// Remove memory entry
bool atlas_remove_memory(const char* key) {
    if (!key) return false;
    
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    bool result = atlas_persistence_remove(g_atlas_ctx, key);
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return result;
}

// Get memory count
size_t atlas_memory_count(void) {
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return 0;
    }
    
    size_t count = atlas_persistence_count(g_atlas_ctx);
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return count;
}

// Manual save
bool atlas_save_memory(void) {
    pthread_mutex_lock(&g_atlas_mutex);
    
    if (!g_atlas_ctx || strlen(g_atlas_filepath) == 0) {
        pthread_mutex_unlock(&g_atlas_mutex);
        return false;
    }
    
    bool result = atlas_persistence_save(g_atlas_ctx, g_atlas_filepath);
    
    pthread_mutex_unlock(&g_atlas_mutex);
    return result;
}

// Set auto-save enabled/disabled
void atlas_set_auto_save(bool enabled) {
    g_auto_save_enabled = enabled;
}

// Get current file path
const char* atlas_get_filepath(void) {
    return g_atlas_filepath;
}

} // extern "C"

// Signal handlers for graceful shutdown
static void atlas_signal_handler(int signal) {
    printf("\nATLAS: Received signal %d, saving memory and shutting down...\n", signal);
    g_shutdown_requested = true;
    atlas_shutdown();
    exit(0);
}

// Register signal handlers  
extern "C" void atlas_register_signal_handlers(void) {
    signal(SIGINT, atlas_signal_handler);
    signal(SIGTERM, atlas_signal_handler);
    signal(SIGUSR1, [](int) { 
        printf("ATLAS: Manual save triggered\n");
        atlas_save_memory();
    });
}

// Demo/CLI interface for testing
static void print_help(void) {
    printf("ATLAS Memory Persistence Demo\n");
    printf("Commands:\n");
    printf("  store <key> <content> [type] - Store memory entry\n");
    printf("  get <key>                    - Retrieve memory entry\n");
    printf("  has <key>                    - Check if key exists\n");
    printf("  remove <key>                 - Remove memory entry\n");
    printf("  list                         - List all keys\n");
    printf("  count                        - Show entry count\n");
    printf("  save                         - Manual save to file\n");
    printf("  info                         - Show file information\n");
    printf("  help                         - Show this help\n");
    printf("  quit                         - Exit demo\n");
}

static bool list_callback(const char* key, const void* data, size_t data_size, 
                         uint32_t type, void* user_data) {
    const char* type_str = "unknown";
    switch (type) {
        case 1: type_str = "text"; break;
        case 2: type_str = "conversation"; break;
        case 3: type_str = "context"; break;
        case 4: type_str = "metadata"; break;
    }
    
    printf("  %-20s [%s] %zu bytes\n", key, type_str, data_size);
    return true;  // Continue iteration
}

extern "C" void run_atlas_demo(const char* model_path) {
    printf("Starting ATLAS Memory Persistence Demo\n");
    printf("Model path: %s\n", model_path ? model_path : "(none)");
    
    // Initialize ATLAS
    if (!atlas_initialize(model_path, nullptr)) {
        printf("Failed to initialize ATLAS persistence\n");
        return;
    }
    
    // Register signal handlers
    atlas_register_signal_handlers();
    
    printf("Atlas file: %s\n", atlas_get_filepath());
    printf("Current entries: %zu\n", atlas_memory_count());
    printf("Type 'help' for commands.\n\n");
    
    char line[1024];
    char command[64], key[256], content[512], type[32];
    
    while (true) {
        printf("atlas> ");
        fflush(stdout);
        
        if (!fgets(line, sizeof(line), stdin)) {
            break;
        }
        
        // Remove trailing newline
        line[strcspn(line, "\n")] = '\0';
        
        if (strlen(line) == 0) continue;
        
        // Parse command
        int args = sscanf(line, "%63s %255s %511s %31s", command, key, content, type);
        
        if (strcmp(command, "help") == 0) {
            print_help();
        }
        else if (strcmp(command, "quit") == 0 || strcmp(command, "exit") == 0) {
            break;
        }
        else if (strcmp(command, "store") == 0 && args >= 3) {
            const char* entry_type = (args >= 4) ? type : "text";
            if (atlas_store_memory(key, content, entry_type)) {
                printf("Stored: %s -> %s (%s)\n", key, content, entry_type);
            } else {
                printf("Failed to store memory entry\n");
            }
        }
        else if (strcmp(command, "get") == 0 && args >= 2) {
            char* retrieved_content = nullptr;
            char* retrieved_type = nullptr;
            if (atlas_retrieve_memory(key, &retrieved_content, &retrieved_type)) {
                printf("Retrieved: %s -> %s (%s)\n", key, retrieved_content, 
                       retrieved_type ? retrieved_type : "unknown");
                free(retrieved_content);
                free(retrieved_type);
            } else {
                printf("Key not found: %s\n", key);
            }
        }
        else if (strcmp(command, "has") == 0 && args >= 2) {
            printf("Key '%s': %s\n", key, atlas_has_memory(key) ? "exists" : "not found");
        }
        else if (strcmp(command, "remove") == 0 && args >= 2) {
            if (atlas_remove_memory(key)) {
                printf("Removed: %s\n", key);
            } else {
                printf("Failed to remove: %s\n", key);
            }
        }
        else if (strcmp(command, "list") == 0) {
            printf("Memory entries (%zu total):\n", atlas_memory_count());
            if (g_atlas_ctx) {
                atlas_persistence_foreach(g_atlas_ctx, list_callback, nullptr);
            }
        }
        else if (strcmp(command, "count") == 0) {
            printf("Entry count: %zu\n", atlas_memory_count());
        }
        else if (strcmp(command, "save") == 0) {
            if (atlas_save_memory()) {
                printf("Memory saved to %s\n", atlas_get_filepath());
            } else {
                printf("Failed to save memory\n");
            }
        }
        else if (strcmp(command, "info") == 0) {
            printf("ATLAS file: %s\n", atlas_get_filepath());
            printf("Entries: %zu\n", atlas_memory_count());
            printf("File exists: %s\n", 
                   atlas_persistence_exists(atlas_get_filepath()) ? "yes" : "no");
        }
        else {
            printf("Unknown command: %s (type 'help' for commands)\n", command);
        }
    }
    
    printf("\nShutting down...\n");
    atlas_shutdown();
}