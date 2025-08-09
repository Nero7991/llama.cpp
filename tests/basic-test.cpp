#include "../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <unistd.h>

int main() {
    printf("=== Basic ATLAS Persistence Test ===\n");
    
    // Test 1: Create context
    printf("Test 1: Creating persistence context... ");
    atlas_persistence_t* ctx = atlas_persistence_create();
    if (!ctx) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    
    // Test 2: Set data
    printf("Test 2: Setting data... ");
    const char* key = "test_key";
    const char* value = "test_value";
    if (!atlas_persistence_set(ctx, key, value, strlen(value), 1)) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    
    // Test 3: Get data
    printf("Test 3: Getting data... ");
    void* retrieved_data;
    size_t data_size;
    uint32_t type;
    if (!atlas_persistence_get(ctx, key, &retrieved_data, &data_size, &type)) {
        printf("FAILED\n");
        return 1;
    }
    
    if (data_size != strlen(value) || type != 1 || memcmp(retrieved_data, value, data_size) != 0) {
        printf("FAILED (data mismatch)\n");
        return 1;
    }
    free(retrieved_data);
    printf("PASSED\n");
    
    // Test 4: Save to file
    printf("Test 4: Saving to file... ");
    const char* test_file = "/tmp/basic_test.atlas";
    if (!atlas_persistence_save(ctx, test_file)) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    
    // Test 5: Load from file
    printf("Test 5: Loading from file... ");
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    if (!atlas_persistence_load(ctx2, test_file)) {
        printf("FAILED\n");
        return 1;
    }
    
    if (atlas_persistence_count(ctx2) != 1) {
        printf("FAILED (wrong count)\n");
        return 1;
    }
    printf("PASSED\n");
    
    // Test 6: Verify loaded data
    printf("Test 6: Verifying loaded data... ");
    void* loaded_data;
    size_t loaded_size;
    uint32_t loaded_type;
    if (!atlas_persistence_get(ctx2, key, &loaded_data, &loaded_size, &loaded_type)) {
        printf("FAILED\n");
        return 1;
    }
    
    if (loaded_size != strlen(value) || loaded_type != 1 || memcmp(loaded_data, value, loaded_size) != 0) {
        printf("FAILED (data mismatch after load)\n");
        return 1;
    }
    free(loaded_data);
    printf("PASSED\n");
    
    // Cleanup
    atlas_persistence_free(ctx);
    atlas_persistence_free(ctx2);
    unlink(test_file);
    
    printf("\n=== All Basic Tests PASSED! ===\n");
    printf("ATLAS Persistence Core Functionality: VERIFIED\n");
    
    return 0;
}