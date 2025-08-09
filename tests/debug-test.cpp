#include "../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>

int main() {
    printf("=== Debug ATLAS Test ===\n");
    
    // Test only create/set/get without file operations
    printf("Creating context... ");
    atlas_persistence_t* ctx = atlas_persistence_create();
    if (!ctx) {
        printf("FAILED - could not create context\n");
        return 1;
    }
    printf("OK\n");
    
    printf("Setting data... ");
    if (!atlas_persistence_set(ctx, "test", "data", 4, 1)) {
        printf("FAILED - could not set data\n");
        return 1;
    }
    printf("OK\n");
    
    printf("Getting data... ");
    void* data;
    size_t size;
    uint32_t type;
    if (!atlas_persistence_get(ctx, "test", &data, &size, &type)) {
        printf("FAILED - could not get data\n");
        return 1;
    }
    printf("OK (size: %zu, type: %u)\n", size, type);
    free(data);
    
    printf("Freeing context... ");
    atlas_persistence_free(ctx);
    printf("OK\n");
    
    printf("All basic operations work!\n");
    return 0;
}