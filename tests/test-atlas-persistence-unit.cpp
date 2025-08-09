#include "../include/atlas-persistence.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>

// Simple test framework
#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    printf("Running " #name "... "); \
    test_##name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define EXPECT_TRUE(expr) do { \
    if (!(expr)) { \
        printf("FAILED: %s at line %d\n", #expr, __LINE__); \
        exit(1); \
    } \
} while(0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))
#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))
#define EXPECT_NE(a, b) EXPECT_TRUE((a) != (b))
#define EXPECT_NULL(ptr) EXPECT_TRUE((ptr) == nullptr)
#define EXPECT_NOT_NULL(ptr) EXPECT_TRUE((ptr) != nullptr)

static int tests_passed = 0;

TEST(persistence_create_and_free) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    EXPECT_EQ(atlas_persistence_count(ctx), 0);
    atlas_persistence_free(ctx);
}

TEST(persistence_set_and_get) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    const char* key = "test_key";
    const char* value = "test_value";
    
    // Set entry
    EXPECT_TRUE(atlas_persistence_set(ctx, key, value, strlen(value), 1));
    EXPECT_EQ(atlas_persistence_count(ctx), 1);
    
    // Get entry
    void* retrieved_data;
    size_t data_size;
    uint32_t type;
    EXPECT_TRUE(atlas_persistence_get(ctx, key, &retrieved_data, &data_size, &type));
    EXPECT_EQ(data_size, strlen(value));
    EXPECT_EQ(type, 1);
    EXPECT_EQ(memcmp(retrieved_data, value, data_size), 0);
    
    free(retrieved_data);
    atlas_persistence_free(ctx);
}

TEST(persistence_has_and_remove) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    const char* key = "test_key";
    const char* value = "test_value";
    
    // Initially not present
    EXPECT_FALSE(atlas_persistence_has(ctx, key));
    
    // Add entry
    EXPECT_TRUE(atlas_persistence_set(ctx, key, value, strlen(value), 1));
    EXPECT_TRUE(atlas_persistence_has(ctx, key));
    EXPECT_EQ(atlas_persistence_count(ctx), 1);
    
    // Remove entry
    EXPECT_TRUE(atlas_persistence_remove(ctx, key));
    EXPECT_FALSE(atlas_persistence_has(ctx, key));
    EXPECT_EQ(atlas_persistence_count(ctx), 0);
    
    atlas_persistence_free(ctx);
}

TEST(persistence_multiple_entries) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    // Add multiple entries
    for (int i = 0; i < 100; i++) {
        char key[32], value[64];
        snprintf(key, sizeof(key), "key_%d", i);
        snprintf(value, sizeof(value), "value_%d_data", i);
        
        EXPECT_TRUE(atlas_persistence_set(ctx, key, value, strlen(value), i % 5));
    }
    
    EXPECT_EQ(atlas_persistence_count(ctx), 100);
    
    // Verify all entries
    for (int i = 0; i < 100; i++) {
        char key[32], expected_value[64];
        snprintf(key, sizeof(key), "key_%d", i);
        snprintf(expected_value, sizeof(expected_value), "value_%d_data", i);
        
        EXPECT_TRUE(atlas_persistence_has(ctx, key));
        
        void* data;
        size_t size;
        uint32_t type;
        EXPECT_TRUE(atlas_persistence_get(ctx, key, &data, &size, &type));
        EXPECT_EQ(size, strlen(expected_value));
        EXPECT_EQ(type, i % 5);
        EXPECT_EQ(memcmp(data, expected_value, size), 0);
        free(data);
    }
    
    atlas_persistence_free(ctx);
}

TEST(persistence_save_and_load) {
    const char* test_file = "/tmp/test_atlas.atlas";
    
    // Create and populate context
    atlas_persistence_t* ctx1 = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx1);
    
    const char* key1 = "session_data";
    const char* value1 = "persistent_session_information";
    const char* key2 = "user_prefs";
    const char* value2 = "user_preference_data";
    
    EXPECT_TRUE(atlas_persistence_set(ctx1, key1, value1, strlen(value1), 2));
    EXPECT_TRUE(atlas_persistence_set(ctx1, key2, value2, strlen(value2), 3));
    EXPECT_EQ(atlas_persistence_count(ctx1), 2);
    
    // Save to file
    EXPECT_TRUE(atlas_persistence_save(ctx1, test_file));
    atlas_persistence_free(ctx1);
    
    // Load into new context
    atlas_persistence_t* ctx2 = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx2);
    EXPECT_TRUE(atlas_persistence_load(ctx2, test_file));
    EXPECT_EQ(atlas_persistence_count(ctx2), 2);
    
    // Verify data integrity
    void* data;
    size_t size;
    uint32_t type;
    
    EXPECT_TRUE(atlas_persistence_get(ctx2, key1, &data, &size, &type));
    EXPECT_EQ(size, strlen(value1));
    EXPECT_EQ(type, 2);
    EXPECT_EQ(memcmp(data, value1, size), 0);
    free(data);
    
    EXPECT_TRUE(atlas_persistence_get(ctx2, key2, &data, &size, &type));
    EXPECT_EQ(size, strlen(value2));
    EXPECT_EQ(type, 3);
    EXPECT_EQ(memcmp(data, value2, size), 0);
    free(data);
    
    atlas_persistence_free(ctx2);
    unlink(test_file);
}

TEST(persistence_file_format_validation) {
    const char* test_file = "/tmp/test_format.atlas";
    
    // Create and save valid file
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    EXPECT_TRUE(atlas_persistence_set(ctx, "test", "data", 4, 1));
    EXPECT_TRUE(atlas_persistence_save(ctx, test_file));
    atlas_persistence_free(ctx);
    
    // Validate file format
    EXPECT_TRUE(atlas_persistence_exists(test_file));
    EXPECT_TRUE(atlas_persistence_validate(test_file));
    
    // Test corrupted file (modify magic header)
    FILE* file = fopen(test_file, "r+b");
    EXPECT_NOT_NULL(file);
    uint32_t bad_magic = 0x12345678;
    fwrite(&bad_magic, sizeof(bad_magic), 1, file);
    fclose(file);
    
    EXPECT_FALSE(atlas_persistence_validate(test_file));
    
    unlink(test_file);
}

TEST(persistence_error_handling) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    // Test invalid parameters
    EXPECT_FALSE(atlas_persistence_set(nullptr, "key", "data", 4, 1));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_INVALID_PARAM);
    
    EXPECT_FALSE(atlas_persistence_set(ctx, nullptr, "data", 4, 1));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_INVALID_PARAM);
    
    EXPECT_FALSE(atlas_persistence_set(ctx, "key", nullptr, 4, 1));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_INVALID_PARAM);
    
    // Test key too long
    char long_key[300];
    memset(long_key, 'a', sizeof(long_key) - 1);
    long_key[sizeof(long_key) - 1] = '\0';
    EXPECT_FALSE(atlas_persistence_set(ctx, long_key, "data", 4, 1));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_KEY_TOO_LONG);
    
    // Test key not found
    void* data;
    size_t size;
    uint32_t type;
    EXPECT_FALSE(atlas_persistence_get(ctx, "nonexistent", &data, &size, &type));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_KEY_NOT_FOUND);
    
    // Test file not found
    EXPECT_FALSE(atlas_persistence_load(ctx, "/nonexistent/path/file.atlas"));
    EXPECT_EQ(atlas_get_last_error(), ATLAS_ERROR_FILE_NOT_FOUND);
    
    atlas_persistence_free(ctx);
}

TEST(persistence_update_existing) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    const char* key = "update_test";
    const char* value1 = "original_value";
    const char* value2 = "updated_value";
    
    // Set initial value
    EXPECT_TRUE(atlas_persistence_set(ctx, key, value1, strlen(value1), 1));
    EXPECT_EQ(atlas_persistence_count(ctx), 1);
    
    // Update with new value
    EXPECT_TRUE(atlas_persistence_set(ctx, key, value2, strlen(value2), 2));
    EXPECT_EQ(atlas_persistence_count(ctx), 1);  // Count shouldn't change
    
    // Verify updated value
    void* data;
    size_t size;
    uint32_t type;
    EXPECT_TRUE(atlas_persistence_get(ctx, key, &data, &size, &type));
    EXPECT_EQ(size, strlen(value2));
    EXPECT_EQ(type, 2);
    EXPECT_EQ(memcmp(data, value2, size), 0);
    
    free(data);
    atlas_persistence_free(ctx);
}

TEST(persistence_auto_filename) {
    char buffer[512];
    
    // Test basic model path
    const char* filename1 = atlas_get_auto_filename("test-model.gguf", buffer);
    EXPECT_NOT_NULL(filename1);
    EXPECT_EQ(strcmp(filename1, "test-model_memory.atlas"), 0);
    
    // Test with path
    const char* filename2 = atlas_get_auto_filename("/path/to/llama-7b.gguf", buffer);
    EXPECT_NOT_NULL(filename2);
    EXPECT_EQ(strcmp(filename2, "llama-7b_memory.atlas"), 0);
    
    // Test without .gguf extension
    const char* filename3 = atlas_get_auto_filename("model", buffer);
    EXPECT_NOT_NULL(filename3);
    EXPECT_EQ(strcmp(filename3, "model_memory.atlas"), 0);
}

TEST(persistence_binary_format) {
    const char* test_file = "/tmp/test_binary.atlas";
    
    // Create file with known data
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    EXPECT_TRUE(atlas_persistence_set(ctx, "binary_test", "ATLS_TEST_DATA", 14, 99));
    EXPECT_TRUE(atlas_persistence_save(ctx, test_file));
    atlas_persistence_free(ctx);
    
    // Read and verify binary format manually
    FILE* file = fopen(test_file, "rb");
    EXPECT_NOT_NULL(file);
    
    // Check magic number
    uint32_t magic;
    EXPECT_EQ(fread(&magic, sizeof(magic), 1, file), 1);
    EXPECT_EQ(magic, 0x534C5441);  // "ATLS" in little endian
    
    // Check version
    uint32_t version;
    EXPECT_EQ(fread(&version, sizeof(version), 1, file), 1);
    EXPECT_EQ(version, 1);
    
    fclose(file);
    unlink(test_file);
}

TEST(persistence_crc32_function) {
    const char* test_data = "Hello, ATLAS!";
    uint32_t crc1 = atlas_crc32(test_data, strlen(test_data));
    uint32_t crc2 = atlas_crc32(test_data, strlen(test_data));
    
    // Same data should produce same CRC
    EXPECT_EQ(crc1, crc2);
    
    // Different data should (likely) produce different CRC
    const char* different_data = "Hello, WORLD!";
    uint32_t crc3 = atlas_crc32(different_data, strlen(different_data));
    EXPECT_NE(crc1, crc3);
    
    // Empty data should not crash
    uint32_t crc4 = atlas_crc32("", 0);
    (void)crc4;  // Just verify it doesn't crash
}

// Callback for foreach test
static int foreach_count = 0;
static bool foreach_callback(const char* key, const void* data, size_t data_size, uint32_t type, void* user_data) {
    EXPECT_NOT_NULL(key);
    EXPECT_NOT_NULL(data);
    EXPECT_TRUE(data_size > 0);
    foreach_count++;
    return true;  // Continue iteration
}

TEST(persistence_foreach) {
    atlas_persistence_t* ctx = atlas_persistence_create();
    EXPECT_NOT_NULL(ctx);
    
    // Add test entries
    EXPECT_TRUE(atlas_persistence_set(ctx, "key1", "data1", 5, 1));
    EXPECT_TRUE(atlas_persistence_set(ctx, "key2", "data2", 5, 2));
    EXPECT_TRUE(atlas_persistence_set(ctx, "key3", "data3", 5, 3));
    
    // Test foreach iteration
    foreach_count = 0;
    atlas_persistence_foreach(ctx, foreach_callback, nullptr);
    EXPECT_EQ(foreach_count, 3);
    
    atlas_persistence_free(ctx);
}

int main() {
    printf("=== ATLAS Persistence Unit Tests ===\n");
    
    RUN_TEST(persistence_create_and_free);
    RUN_TEST(persistence_set_and_get);
    RUN_TEST(persistence_has_and_remove);
    RUN_TEST(persistence_multiple_entries);
    RUN_TEST(persistence_save_and_load);
    RUN_TEST(persistence_file_format_validation);
    RUN_TEST(persistence_error_handling);
    RUN_TEST(persistence_update_existing);
    RUN_TEST(persistence_auto_filename);
    RUN_TEST(persistence_binary_format);
    RUN_TEST(persistence_crc32_function);
    RUN_TEST(persistence_foreach);
    
    printf("\n=== Test Results ===\n");
    printf("All %d tests PASSED!\n", tests_passed);
    printf("ATLAS Persistence Unit Tests: SUCCESS\n");
    
    return 0;
}