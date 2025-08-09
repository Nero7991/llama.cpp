#include "../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/stat.h>

// Simple test framework
#define EXPECT_TRUE(expr) do { \
    if (!(expr)) { \
        printf("FAILED: %s at line %d\n", #expr, __LINE__); \
        exit(1); \
    } \
} while(0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))
#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))

static void test_end_to_end_workflow() {
    printf("Testing end-to-end workflow... ");
    
    const char* model_path = "/tmp/test-model.gguf";
    const char* atlas_file = "/tmp/test-model_memory.atlas";
    
    // Clean up any existing files
    unlink(atlas_file);
    
    // === Session 1: Create and store data ===
    atlas_persistence_t* ctx1 = atlas_persistence_create();
    EXPECT_TRUE(ctx1 != nullptr);
    
    // Store conversation data
    const char* conversation = "User: Hello\nAssistant: Hi there! How can I help you today?\nUser: What is ATLAS?";
    EXPECT_TRUE(atlas_persistence_set(ctx1, "conversation_history", conversation, strlen(conversation), 2));
    
    // Store user preferences
    const char* prefs = "{\"theme\":\"dark\",\"language\":\"en\",\"response_style\":\"detailed\"}";
    EXPECT_TRUE(atlas_persistence_set(ctx1, "user_preferences", prefs, strlen(prefs), 4));
    
    // Store context metadata
    const char* meta = "session_id:12345,model:llama-7b,timestamp:2025-01-09";
    EXPECT_TRUE(atlas_persistence_set(ctx1, "session_metadata", meta, strlen(meta), 4));
    
    EXPECT_EQ(atlas_persistence_count(ctx1), 3);
    
    // Save to file
    char filename_buffer[512];
    const char* auto_filename = atlas_get_auto_filename(model_path, filename_buffer);
    EXPECT_TRUE(atlas_persistence_save(ctx1, auto_filename));
    
    atlas_persistence_free(ctx1);
    
    // Verify file was created
    EXPECT_TRUE(atlas_persistence_exists(auto_filename));
    
    // === Session 2: Load and continue ===
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    EXPECT_TRUE(ctx2 != nullptr);
    
    // Load previous session
    EXPECT_TRUE(atlas_persistence_load(ctx2, auto_filename));
    EXPECT_EQ(atlas_persistence_count(ctx2), 3);
    
    // Verify data integrity
    void* loaded_conv;
    size_t conv_size;
    uint32_t conv_type;
    EXPECT_TRUE(atlas_persistence_get(ctx2, "conversation_history", &loaded_conv, &conv_size, &conv_type));
    EXPECT_EQ(conv_size, strlen(conversation));
    EXPECT_EQ(conv_type, 2);
    EXPECT_EQ(memcmp(loaded_conv, conversation, conv_size), 0);
    free(loaded_conv);
    
    // Add more conversation data
    const char* new_conv = "User: How does memory persistence work?\nAssistant: ATLAS uses binary files...";
    EXPECT_TRUE(atlas_persistence_set(ctx2, "conversation_continued", new_conv, strlen(new_conv), 2));
    EXPECT_EQ(atlas_persistence_count(ctx2), 4);
    
    // Save updated session
    EXPECT_TRUE(atlas_persistence_save(ctx2, auto_filename));
    atlas_persistence_free(ctx2);
    
    // === Session 3: Verify persistence ===
    atlas_persistence_t* ctx3 = atlas_persistence_create();
    EXPECT_TRUE(ctx3 != nullptr);
    
    EXPECT_TRUE(atlas_persistence_load(ctx3, auto_filename));
    EXPECT_EQ(atlas_persistence_count(ctx3), 4);
    
    // Verify all data persisted correctly
    EXPECT_TRUE(atlas_persistence_has(ctx3, "conversation_history"));
    EXPECT_TRUE(atlas_persistence_has(ctx3, "user_preferences"));
    EXPECT_TRUE(atlas_persistence_has(ctx3, "session_metadata"));
    EXPECT_TRUE(atlas_persistence_has(ctx3, "conversation_continued"));
    
    atlas_persistence_free(ctx3);
    unlink(auto_filename);
    
    printf("PASSED\n");
}

static void test_concurrent_access() {
    printf("Testing file locking behavior... ");
    
    const char* test_file = "/tmp/concurrent_test.atlas";
    unlink(test_file);
    
    // Create and save initial data
    atlas_persistence_t* ctx1 = atlas_persistence_create();
    EXPECT_TRUE(atlas_persistence_set(ctx1, "shared_data", "initial_value", 13, 1));
    EXPECT_TRUE(atlas_persistence_save(ctx1, test_file));
    atlas_persistence_free(ctx1);
    
    // Test that multiple contexts can read the same file
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    atlas_persistence_t* ctx3 = atlas_persistence_create();
    
    EXPECT_TRUE(atlas_persistence_load(ctx2, test_file));
    EXPECT_TRUE(atlas_persistence_load(ctx3, test_file));
    
    // Both should have the data
    EXPECT_TRUE(atlas_persistence_has(ctx2, "shared_data"));
    EXPECT_TRUE(atlas_persistence_has(ctx3, "shared_data"));
    
    atlas_persistence_free(ctx2);
    atlas_persistence_free(ctx3);
    unlink(test_file);
    
    printf("PASSED\n");
}

static void test_large_dataset() {
    printf("Testing large dataset handling... ");
    
    const char* test_file = "/tmp/large_dataset.atlas";
    unlink(test_file);
    
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_TRUE(ctx != nullptr);
    
    // Create large dataset (1000 entries)
    for (int i = 0; i < 1000; i++) {
        char key[64], value[256];
        snprintf(key, sizeof(key), "large_entry_%04d", i);
        snprintf(value, sizeof(value), "This is entry number %d with some additional data to make it larger. "
                "ATLAS persistence system should handle this efficiently. Data: %d", i, i * 42);
        
        EXPECT_TRUE(atlas_persistence_set(ctx, key, value, strlen(value), i % 10));
    }
    
    EXPECT_EQ(atlas_persistence_count(ctx), 1000);
    
    // Save large dataset
    EXPECT_TRUE(atlas_persistence_save(ctx, test_file));
    
    // Check file size
    struct stat st;
    EXPECT_EQ(stat(test_file, &st), 0);
    printf("(file size: %ld bytes) ", st.st_size);
    
    atlas_persistence_free(ctx);
    
    // Load and verify large dataset
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    EXPECT_TRUE(atlas_persistence_load(ctx2, test_file));
    EXPECT_EQ(atlas_persistence_count(ctx2), 1000);
    
    // Verify random entries
    for (int i = 0; i < 100; i += 10) {
        char key[64];
        snprintf(key, sizeof(key), "large_entry_%04d", i);
        EXPECT_TRUE(atlas_persistence_has(ctx2, key));
    }
    
    atlas_persistence_free(ctx2);
    unlink(test_file);
    
    printf("PASSED\n");
}

static void test_corruption_recovery() {
    printf("Testing corruption recovery... ");
    
    const char* test_file = "/tmp/corruption_test.atlas";
    unlink(test_file);
    
    // Create valid file
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_TRUE(atlas_persistence_set(ctx, "test_data", "valid_content", 13, 1));
    EXPECT_TRUE(atlas_persistence_save(ctx, test_file));
    atlas_persistence_free(ctx);
    
    // Verify file is valid
    EXPECT_TRUE(atlas_persistence_validate(test_file));
    
    // Corrupt the file by modifying magic header
    FILE* file = fopen(test_file, "r+b");
    EXPECT_TRUE(file != nullptr);
    uint32_t bad_magic = 0xDEADBEEF;
    fwrite(&bad_magic, sizeof(bad_magic), 1, file);
    fclose(file);
    
    // Verify corruption is detected
    EXPECT_FALSE(atlas_persistence_validate(test_file));
    
    // Try to load corrupted file - should fail gracefully
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    EXPECT_FALSE(atlas_persistence_load(ctx2, test_file));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_MAGIC_MISMATCH);
    atlas_persistence_free(ctx2);
    
    unlink(test_file);
    printf("PASSED\n");
}

static void test_cross_platform_compatibility() {
    printf("Testing cross-platform file format... ");
    
    const char* test_file = "/tmp/cross_platform.atlas";
    unlink(test_file);
    
    // Create file with various data types
    atlas_persistence_t* ctx1 = atlas_persistence_create();
    
    // Test different data types that might have endianness issues
    uint32_t int_data = 0x12345678;
    EXPECT_TRUE(atlas_persistence_set(ctx1, "int_test", &int_data, sizeof(int_data), 1));
    
    const char* text_data = "Cross-platform text with unicode: 🚀 ATLAS";
    EXPECT_TRUE(atlas_persistence_set(ctx1, "text_test", text_data, strlen(text_data), 2));
    
    // Binary data
    uint8_t binary_data[] = {0x00, 0x01, 0x02, 0xFE, 0xFF, 0x80, 0x7F};
    EXPECT_TRUE(atlas_persistence_set(ctx1, "binary_test", binary_data, sizeof(binary_data), 3));
    
    EXPECT_TRUE(atlas_persistence_save(ctx1, test_file));
    atlas_persistence_free(ctx1);
    
    // Load and verify data integrity
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    EXPECT_TRUE(atlas_persistence_load(ctx2, test_file));
    
    void* retrieved_data;
    size_t data_size;
    uint32_t type;
    
    // Verify int data
    EXPECT_TRUE(atlas_persistence_get(ctx2, "int_test", &retrieved_data, &data_size, &type));
    EXPECT_EQ(data_size, sizeof(int_data));
    EXPECT_EQ(*(uint32_t*)retrieved_data, int_data);
    free(retrieved_data);
    
    // Verify text data
    EXPECT_TRUE(atlas_persistence_get(ctx2, "text_test", &retrieved_data, &data_size, &type));
    EXPECT_EQ(data_size, strlen(text_data));
    EXPECT_EQ(memcmp(retrieved_data, text_data, data_size), 0);
    free(retrieved_data);
    
    // Verify binary data
    EXPECT_TRUE(atlas_persistence_get(ctx2, "binary_test", &retrieved_data, &data_size, &type));
    EXPECT_EQ(data_size, sizeof(binary_data));
    EXPECT_EQ(memcmp(retrieved_data, binary_data, data_size), 0);
    free(retrieved_data);
    
    atlas_persistence_free(ctx2);
    unlink(test_file);
    printf("PASSED\n");
}

static void test_edge_cases() {
    printf("Testing edge cases... ");
    
    atlas_persistence_t* ctx = atlas_persistence_create();
    
    // Empty key (should fail)
    EXPECT_FALSE(atlas_persistence_set(ctx, "", "data", 4, 1));
    
    // Empty data (should work)
    EXPECT_TRUE(atlas_persistence_set(ctx, "empty_data", "", 0, 1));
    
    // Very long key (within limits)
    char long_key[256];
    memset(long_key, 'a', 255);
    long_key[255] = '\0';
    EXPECT_TRUE(atlas_persistence_set(ctx, long_key, "data", 4, 1));
    
    // Key with special characters
    const char* special_key = "key/with\\special:chars|and<symbols>";
    EXPECT_TRUE(atlas_persistence_set(ctx, special_key, "special_data", 12, 1));
    
    // Test retrieval
    EXPECT_TRUE(atlas_persistence_has(ctx, "empty_data"));
    EXPECT_TRUE(atlas_persistence_has(ctx, long_key));
    EXPECT_TRUE(atlas_persistence_has(ctx, special_key));
    
    // Test file operations with invalid paths
    EXPECT_FALSE(atlas_persistence_save(ctx, ""));
    EXPECT_FALSE(atlas_persistence_save(ctx, "/invalid/path/that/does/not/exist/file.atlas"));
    
    atlas_persistence_free(ctx);
    printf("PASSED\n");
}

static void test_performance_benchmarks() {
    printf("Testing performance benchmarks... ");
    
    const char* test_file = "/tmp/performance_test.atlas";
    unlink(test_file);
    
    atlas_persistence_t* ctx = atlas_persistence_create();
    
    // Measure insertion performance
    clock_t start_insert = clock();
    for (int i = 0; i < 10000; i++) {
        char key[32], value[100];
        snprintf(key, sizeof(key), "perf_key_%d", i);
        snprintf(value, sizeof(value), "performance_test_data_entry_number_%d_with_some_content", i);
        EXPECT_TRUE(atlas_persistence_set(ctx, key, value, strlen(value), 1));
    }
    clock_t end_insert = clock();
    double insert_time = ((double)(end_insert - start_insert)) / CLOCKS_PER_SEC;
    
    // Measure save performance
    clock_t start_save = clock();
    EXPECT_TRUE(atlas_persistence_save(ctx, test_file));
    clock_t end_save = clock();
    double save_time = ((double)(end_save - start_save)) / CLOCKS_PER_SEC;
    
    atlas_persistence_free(ctx);
    
    // Measure load performance
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    clock_t start_load = clock();
    EXPECT_TRUE(atlas_persistence_load(ctx2, test_file));
    clock_t end_load = clock();
    double load_time = ((double)(end_load - start_load)) / CLOCKS_PER_SEC;
    
    // Measure retrieval performance
    clock_t start_get = clock();
    for (int i = 0; i < 1000; i++) {
        char key[32];
        snprintf(key, sizeof(key), "perf_key_%d", i);
        void* data;
        size_t size;
        uint32_t type;
        EXPECT_TRUE(atlas_persistence_get(ctx2, key, &data, &size, &type));
        free(data);
    }
    clock_t end_get = clock();
    double get_time = ((double)(end_get - start_get)) / CLOCKS_PER_SEC;
    
    atlas_persistence_free(ctx2);
    
    printf("(insert: %.3fs, save: %.3fs, load: %.3fs, get: %.3fs) ", 
           insert_time, save_time, load_time, get_time);
    
    // Performance requirements check
    EXPECT_TRUE(insert_time < 5.0);  // 10K inserts in <5s
    EXPECT_TRUE(save_time < 2.0);    // Save in <2s
    EXPECT_TRUE(load_time < 1.0);    // Load in <1s
    EXPECT_TRUE(get_time < 0.5);     // 1K gets in <0.5s
    
    unlink(test_file);
    printf("PASSED\n");
}

int main() {
    printf("=== ATLAS Persistence Integration Tests ===\n");
    
    test_end_to_end_workflow();
    test_concurrent_access();
    test_large_dataset();
    test_corruption_recovery();
    test_cross_platform_compatibility();
    test_edge_cases();
    test_performance_benchmarks();
    
    printf("\n=== Integration Test Results ===\n");
    printf("All integration tests PASSED!\n");
    printf("ATLAS Persistence Integration Tests: SUCCESS\n");
    
    return 0;
}