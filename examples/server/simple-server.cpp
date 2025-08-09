#include "../../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>

// Simple demonstration without complex threading or signal handling
int main(int argc, char** argv) {
    if (argc < 3 || strcmp(argv[1], "-m") != 0) {
        printf("Usage: %s -m model_path\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[2];
    
    printf("=== Simple ATLAS Persistence Demo ===\n");
    printf("Model: %s\n", model_path);
    
    // Create persistence context
    atlas_persistence_t* ctx = atlas_persistence_create();
    if (!ctx) {
        printf("Failed to create persistence context\n");
        return 1;
    }
    
    // Generate atlas filename
    char atlas_filename[512];
    atlas_get_auto_filename(model_path, atlas_filename);
    printf("Atlas file: %s\n", atlas_filename);
    
    // Try to load existing file
    if (atlas_persistence_exists(atlas_filename)) {
        if (atlas_persistence_load(ctx, atlas_filename)) {
            printf("Loaded existing memory (%zu entries)\n", 
                   atlas_persistence_count(ctx));
        } else {
            printf("Failed to load existing file\n");
        }
    } else {
        printf("Starting with fresh memory\n");
    }
    
    // Store a test entry
    const char* test_key = "demo_entry";
    const char* test_value = "ATLAS Phase 6B demonstration data";
    
    if (atlas_persistence_set(ctx, test_key, test_value, strlen(test_value), 1)) {
        printf("Stored: %s -> %s\n", test_key, test_value);
    } else {
        printf("Failed to store entry\n");
    }
    
    // Show current count
    printf("Current entries: %zu\n", atlas_persistence_count(ctx));
    
    // Save to file
    if (atlas_persistence_save(ctx, atlas_filename)) {
        printf("Saved memory to %s\n", atlas_filename);
    } else {
        printf("Failed to save memory\n");
    }
    
    // Clean up
    atlas_persistence_free(ctx);
    
    printf("Demo completed successfully\n");
    return 0;
}