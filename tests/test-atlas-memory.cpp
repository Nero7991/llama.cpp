// Unit tests for ATLAS memory management
// Tests memory allocation, pool management, and memory optimization

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <memory>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ATLAS memory management types
enum atlas_memory_type {
    ATLAS_MEM_HOST = 0,
    ATLAS_MEM_DEVICE,
    ATLAS_MEM_PINNED,
    ATLAS_MEM_UNIFIED,
    ATLAS_MEM_COUNT
};

struct atlas_memory_block {
    void* ptr;
    size_t size;
    size_t alignment;
    atlas_memory_type type;
    bool is_free;
    size_t offset;  // Offset from pool base
    struct atlas_memory_block* next;
    struct atlas_memory_block* prev;
};

struct atlas_memory_pool {
    void* base_ptr;
    size_t total_size;
    size_t used_size;
    size_t peak_usage;
    size_t block_count;
    atlas_memory_type type;
    atlas_memory_block* free_blocks;
    atlas_memory_block* used_blocks;
    std::mutex pool_mutex;
    
    // Pool statistics
    size_t allocation_count;
    size_t deallocation_count;
    size_t fragmentation_score;
    double utilization_ratio;
};

struct atlas_memory_manager {
    atlas_memory_pool* pools[ATLAS_MEM_COUNT];
    std::atomic<size_t> total_allocated;
    std::atomic<size_t> peak_memory;
    std::mutex manager_mutex;
    
    // Allocation strategies
    enum {
        ATLAS_ALLOC_FIRST_FIT,
        ATLAS_ALLOC_BEST_FIT,
        ATLAS_ALLOC_WORST_FIT
    } allocation_strategy;
    
    // Memory optimization settings
    bool enable_coalescing;
    bool enable_prefetching;
    size_t min_block_size;
    size_t max_fragmentation;
};

// Test constants
constexpr size_t TEST_POOL_SIZE = 64 * 1024 * 1024; // 64MB
constexpr size_t MIN_ALIGNMENT = 32;
constexpr float TEST_TOLERANCE = 1e-5f;

// Global memory manager for testing
static atlas_memory_manager g_memory_manager = {};

// Memory pool management functions
static atlas_memory_pool* create_memory_pool(atlas_memory_type type, size_t size) {
    atlas_memory_pool* pool = new atlas_memory_pool();
    if (!pool) return nullptr;
    
    std::memset(pool, 0, sizeof(*pool));
    
    pool->type = type;
    pool->total_size = size;
    pool->allocation_strategy = g_memory_manager.allocation_strategy;
    
    // Allocate pool memory based on type
    switch (type) {
        case ATLAS_MEM_HOST:
            pool->base_ptr = std::aligned_alloc(MIN_ALIGNMENT, size);
            break;
        case ATLAS_MEM_DEVICE:
            // Mock device allocation
            pool->base_ptr = std::aligned_alloc(MIN_ALIGNMENT, size);
            break;
        case ATLAS_MEM_PINNED:
            // Mock pinned allocation
            pool->base_ptr = std::aligned_alloc(MIN_ALIGNMENT, size);
            break;
        case ATLAS_MEM_UNIFIED:
            // Mock unified allocation
            pool->base_ptr = std::aligned_alloc(MIN_ALIGNMENT, size);
            break;
        default:
            delete pool;
            return nullptr;
    }
    
    if (!pool->base_ptr) {
        delete pool;
        return nullptr;
    }
    
    // Initialize with single free block
    atlas_memory_block* initial_block = new atlas_memory_block();
    initial_block->ptr = pool->base_ptr;
    initial_block->size = size;
    initial_block->alignment = MIN_ALIGNMENT;
    initial_block->type = type;
    initial_block->is_free = true;
    initial_block->offset = 0;
    initial_block->next = nullptr;
    initial_block->prev = nullptr;
    
    pool->free_blocks = initial_block;
    pool->block_count = 1;
    
    return pool;
}

static void destroy_memory_pool(atlas_memory_pool* pool) {
    if (!pool) return;
    
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    // Free all blocks
    atlas_memory_block* block = pool->free_blocks;
    while (block) {
        atlas_memory_block* next = block->next;
        delete block;
        block = next;
    }
    
    block = pool->used_blocks;
    while (block) {
        atlas_memory_block* next = block->next;
        delete block;
        block = next;
    }
    
    // Free pool memory
    if (pool->base_ptr) {
        std::free(pool->base_ptr);
    }
    
    delete pool;
}

static atlas_memory_block* find_free_block(atlas_memory_pool* pool, size_t size, size_t alignment) {
    atlas_memory_block* best_block = nullptr;
    atlas_memory_block* block = pool->free_blocks;
    
    while (block) {
        if (block->is_free && block->size >= size) {
            // Check alignment requirements
            uintptr_t addr = reinterpret_cast<uintptr_t>(block->ptr);
            uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
            size_t padding = aligned_addr - addr;
            
            if (block->size >= size + padding) {
                switch (g_memory_manager.allocation_strategy) {
                    case atlas_memory_manager::ATLAS_ALLOC_FIRST_FIT:
                        return block;
                    case atlas_memory_manager::ATLAS_ALLOC_BEST_FIT:
                        if (!best_block || block->size < best_block->size) {
                            best_block = block;
                        }
                        break;
                    case atlas_memory_manager::ATLAS_ALLOC_WORST_FIT:
                        if (!best_block || block->size > best_block->size) {
                            best_block = block;
                        }
                        break;
                }
            }
        }
        block = block->next;
    }
    
    return best_block;
}

static void* allocate_from_pool(atlas_memory_pool* pool, size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    atlas_memory_block* block = find_free_block(pool, size, alignment);
    if (!block) return nullptr;
    
    // Calculate aligned address
    uintptr_t addr = reinterpret_cast<uintptr_t>(block->ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    size_t padding = aligned_addr - addr;
    
    // Split block if necessary
    if (block->size > size + padding + sizeof(atlas_memory_block)) {
        atlas_memory_block* new_block = new atlas_memory_block();
        new_block->ptr = (char*)block->ptr + size + padding;
        new_block->size = block->size - size - padding;
        new_block->alignment = alignment;
        new_block->type = block->type;
        new_block->is_free = true;
        new_block->offset = block->offset + size + padding;
        new_block->next = block->next;
        new_block->prev = block;
        
        if (block->next) {
            block->next->prev = new_block;
        }
        block->next = new_block;
        pool->block_count++;
        
        block->size = size + padding;
    }
    
    // Mark block as used
    block->is_free = false;
    block->alignment = alignment;
    
    // Move from free list to used list
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        pool->free_blocks = block->next;
    }
    if (block->next) {
        block->next->prev = block->prev;
    }
    
    // Add to used list
    block->next = pool->used_blocks;
    block->prev = nullptr;
    if (pool->used_blocks) {
        pool->used_blocks->prev = block;
    }
    pool->used_blocks = block;
    
    pool->used_size += block->size;
    pool->allocation_count++;
    
    if (pool->used_size > pool->peak_usage) {
        pool->peak_usage = pool->used_size;
    }
    
    return reinterpret_cast<void*>(aligned_addr);
}

static void deallocate_from_pool(atlas_memory_pool* pool, void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    
    // Find block in used list
    atlas_memory_block* block = pool->used_blocks;
    while (block) {
        uintptr_t block_addr = reinterpret_cast<uintptr_t>(block->ptr);
        uintptr_t aligned_addr = (block_addr + block->alignment - 1) & ~(block->alignment - 1);
        
        if (reinterpret_cast<void*>(aligned_addr) == ptr) {
            break;
        }
        block = block->next;
    }
    
    if (!block) return; // Block not found
    
    // Remove from used list
    if (block->prev) {
        block->prev->next = block->next;
    } else {
        pool->used_blocks = block->next;
    }
    if (block->next) {
        block->next->prev = block->prev;
    }
    
    // Mark as free
    block->is_free = true;
    pool->used_size -= block->size;
    pool->deallocation_count++;
    
    // Add to free list (coalescing if enabled)
    if (g_memory_manager.enable_coalescing) {
        // Simple coalescing - merge with adjacent free blocks
        atlas_memory_block* free_block = pool->free_blocks;
        atlas_memory_block* insert_after = nullptr;
        
        while (free_block && free_block->offset < block->offset) {
            // Check if can merge with previous block
            if (free_block->offset + free_block->size == block->offset) {
                free_block->size += block->size;
                delete block;
                pool->block_count--;
                return;
            }
            insert_after = free_block;
            free_block = free_block->next;
        }
        
        // Check if can merge with next block
        if (free_block && block->offset + block->size == free_block->offset) {
            block->size += free_block->size;
            block->next = free_block->next;
            if (free_block->next) {
                free_block->next->prev = block;
            }
            delete free_block;
            pool->block_count--;
        } else {
            block->next = free_block;
            if (free_block) {
                free_block->prev = block;
            }
        }
        
        // Insert in sorted order
        if (insert_after) {
            block->prev = insert_after;
            insert_after->next = block;
        } else {
            block->prev = nullptr;
            pool->free_blocks = block;
        }
    } else {
        // Simple insertion at head
        block->next = pool->free_blocks;
        block->prev = nullptr;
        if (pool->free_blocks) {
            pool->free_blocks->prev = block;
        }
        pool->free_blocks = block;
    }
}

// Memory manager functions
static bool initialize_memory_manager() {
    std::memset(&g_memory_manager, 0, sizeof(g_memory_manager));
    
    g_memory_manager.allocation_strategy = atlas_memory_manager::ATLAS_ALLOC_BEST_FIT;
    g_memory_manager.enable_coalescing = true;
    g_memory_manager.enable_prefetching = false;
    g_memory_manager.min_block_size = 64;
    g_memory_manager.max_fragmentation = 1024;
    
    // Create pools for each memory type
    for (int i = 0; i < ATLAS_MEM_COUNT; i++) {
        g_memory_manager.pools[i] = create_memory_pool(
            static_cast<atlas_memory_type>(i), TEST_POOL_SIZE);
        if (!g_memory_manager.pools[i]) {
            return false;
        }
    }
    
    return true;
}

static void shutdown_memory_manager() {
    for (int i = 0; i < ATLAS_MEM_COUNT; i++) {
        if (g_memory_manager.pools[i]) {
            destroy_memory_pool(g_memory_manager.pools[i]);
            g_memory_manager.pools[i] = nullptr;
        }
    }
}

// Test utility functions
static void generate_random_sizes(std::vector<size_t>& sizes, size_t count) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dis(64, 8192);
    
    sizes.clear();
    for (size_t i = 0; i < count; i++) {
        sizes.push_back(dis(gen));
    }
}

// Test functions

// Test 1: Memory Pool Creation and Destruction
static bool test_memory_pool_creation() {
    printf("Testing memory pool creation and destruction... ");
    
    // Test creating pools for each memory type
    bool success = true;
    
    for (int i = 0; i < ATLAS_MEM_COUNT; i++) {
        atlas_memory_type type = static_cast<atlas_memory_type>(i);
        atlas_memory_pool* pool = create_memory_pool(type, TEST_POOL_SIZE);
        
        success &= (pool != nullptr);
        success &= (pool->type == type);
        success &= (pool->total_size == TEST_POOL_SIZE);
        success &= (pool->used_size == 0);
        success &= (pool->base_ptr != nullptr);
        success &= (pool->free_blocks != nullptr);
        success &= (pool->used_blocks == nullptr);
        success &= (pool->block_count == 1);
        
        destroy_memory_pool(pool);
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 2: Basic Memory Allocation and Deallocation
static bool test_basic_allocation() {
    printf("Testing basic memory allocation and deallocation... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
    
    // Test various allocation sizes
    const size_t test_sizes[] = {64, 256, 1024, 4096, 16384};
    const size_t num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    std::vector<void*> allocations;
    
    // Allocate memory blocks
    for (size_t i = 0; i < num_sizes; i++) {
        void* ptr = allocate_from_pool(pool, test_sizes[i], MIN_ALIGNMENT);
        success &= (ptr != nullptr);
        
        if (ptr) {
            // Check alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            success &= ((addr % MIN_ALIGNMENT) == 0);
            
            // Test memory is writable
            std::memset(ptr, 0x42, test_sizes[i]);
            success &= (((char*)ptr)[0] == 0x42);
            success &= (((char*)ptr)[test_sizes[i]-1] == 0x42);
            
            allocations.push_back(ptr);
        }
    }
    
    // Check pool statistics
    success &= (pool->allocation_count == num_sizes);
    success &= (pool->used_size > 0);
    
    // Deallocate memory blocks
    for (void* ptr : allocations) {
        deallocate_from_pool(pool, ptr);
    }
    
    // Check pool statistics after deallocation
    success &= (pool->deallocation_count == num_sizes);
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 3: Memory Alignment
static bool test_memory_alignment() {
    printf("Testing memory alignment... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
    
    // Test various alignments
    const size_t alignments[] = {16, 32, 64, 128, 256};
    const size_t num_alignments = sizeof(alignments) / sizeof(alignments[0]);
    
    for (size_t i = 0; i < num_alignments; i++) {
        size_t alignment = alignments[i];
        size_t size = 1024;
        
        void* ptr = allocate_from_pool(pool, size, alignment);
        success &= (ptr != nullptr);
        
        if (ptr) {
            // Check alignment
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            success &= ((addr % alignment) == 0);
            
            deallocate_from_pool(pool, ptr);
        }
    }
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 4: Memory Fragmentation and Coalescing
static bool test_fragmentation_coalescing() {
    printf("Testing memory fragmentation and coalescing... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
    
    // Allocate many small blocks
    const size_t block_size = 1024;
    const size_t num_blocks = 100;
    std::vector<void*> allocations;
    
    for (size_t i = 0; i < num_blocks; i++) {
        void* ptr = allocate_from_pool(pool, block_size, MIN_ALIGNMENT);
        success &= (ptr != nullptr);
        if (ptr) {
            allocations.push_back(ptr);
        }
    }
    
    size_t initial_block_count = pool->block_count;
    
    // Deallocate every other block to create fragmentation
    for (size_t i = 0; i < allocations.size(); i += 2) {
        deallocate_from_pool(pool, allocations[i]);
    }
    
    // Count free blocks (fragmentation indicator)
    size_t free_block_count = 0;
    atlas_memory_block* block = pool->free_blocks;
    while (block) {
        free_block_count++;
        block = block->next;
    }
    
    // Should have significant fragmentation
    success &= (free_block_count > 1);
    
    // Deallocate remaining blocks
    for (size_t i = 1; i < allocations.size(); i += 2) {
        deallocate_from_pool(pool, allocations[i]);
    }
    
    // If coalescing is enabled, should have fewer free blocks now
    if (g_memory_manager.enable_coalescing) {
        size_t final_free_count = 0;
        block = pool->free_blocks;
        while (block) {
            final_free_count++;
            block = block->next;
        }
        success &= (final_free_count < free_block_count);
    }
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 5: Multi-threaded Memory Access
static bool test_multithreaded_access() {
    printf("Testing multi-threaded memory access... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
    
    const int num_threads = 4;
    const int allocations_per_thread = 50;
    std::atomic<int> completed_threads{0};
    std::vector<bool> thread_results(num_threads, false);
    
    auto worker_function = [&](int thread_id) {
        std::vector<void*> local_allocations;
        bool local_success = true;
        
        // Perform allocations
        for (int i = 0; i < allocations_per_thread; i++) {
            size_t size = 1024 + (thread_id * 100) + (i * 10);
            void* ptr = allocate_from_pool(pool, size, MIN_ALIGNMENT);
            
            if (ptr) {
                // Write thread-specific pattern
                char pattern = (char)(0x40 + thread_id);
                std::memset(ptr, pattern, size);
                local_allocations.push_back(ptr);
            } else {
                local_success = false;
                break;
            }
        }
        
        // Verify allocations
        for (size_t i = 0; i < local_allocations.size(); i++) {
            void* ptr = local_allocations[i];
            size_t size = 1024 + (thread_id * 100) + (i * 10);
            char expected_pattern = (char)(0x40 + thread_id);
            
            char* data = (char*)ptr;
            if (data[0] != expected_pattern || data[size-1] != expected_pattern) {
                local_success = false;
                break;
            }
        }
        
        // Deallocate
        for (void* ptr : local_allocations) {
            deallocate_from_pool(pool, ptr);
        }
        
        thread_results[thread_id] = local_success;
        completed_threads++;
    };
    
    // Launch worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker_function, i);
    }
    
    // Wait for completion
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Check results
    for (bool result : thread_results) {
        success &= result;
    }
    
    success &= (completed_threads == num_threads);
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 6: Memory Pool Statistics
static bool test_pool_statistics() {
    printf("Testing memory pool statistics... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
    
    // Initial statistics
    success &= (pool->allocation_count == 0);
    success &= (pool->deallocation_count == 0);
    success &= (pool->used_size == 0);
    success &= (pool->peak_usage == 0);
    
    // Perform allocations
    const size_t num_allocations = 10;
    const size_t block_size = 4096;
    std::vector<void*> allocations;
    
    for (size_t i = 0; i < num_allocations; i++) {
        void* ptr = allocate_from_pool(pool, block_size, MIN_ALIGNMENT);
        if (ptr) {
            allocations.push_back(ptr);
        }
    }
    
    // Check statistics after allocations
    success &= (pool->allocation_count == num_allocations);
    success &= (pool->used_size > 0);
    success &= (pool->peak_usage >= pool->used_size);
    
    size_t peak_after_allocs = pool->peak_usage;
    
    // Deallocate half
    size_t half = allocations.size() / 2;
    for (size_t i = 0; i < half; i++) {
        deallocate_from_pool(pool, allocations[i]);
    }
    
    // Peak should remain the same
    success &= (pool->peak_usage == peak_after_allocs);
    success &= (pool->deallocation_count == half);
    
    // Deallocate rest
    for (size_t i = half; i < allocations.size(); i++) {
        deallocate_from_pool(pool, allocations[i]);
    }
    
    success &= (pool->deallocation_count == allocations.size());
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 7: Different Memory Types
static bool test_memory_types() {
    printf("Testing different memory types... ");
    
    if (!initialize_memory_manager()) {
        printf("FAILED - Could not initialize memory manager\n");
        return false;
    }
    
    bool success = true;
    
    // Test allocation from each memory type
    const size_t test_size = 4096;
    
    for (int i = 0; i < ATLAS_MEM_COUNT; i++) {
        atlas_memory_type type = static_cast<atlas_memory_type>(i);
        atlas_memory_pool* pool = g_memory_manager.pools[type];
        
        success &= (pool != nullptr);
        success &= (pool->type == type);
        
        // Test allocation
        void* ptr = allocate_from_pool(pool, test_size, MIN_ALIGNMENT);
        success &= (ptr != nullptr);
        
        if (ptr) {
            // Test memory is usable
            std::memset(ptr, 0x55, test_size);
            success &= (((char*)ptr)[0] == 0x55);
            success &= (((char*)ptr)[test_size-1] == 0x55);
            
            deallocate_from_pool(pool, ptr);
        }
    }
    
    shutdown_memory_manager();
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

// Test 8: Allocation Strategy Comparison
static bool test_allocation_strategies() {
    printf("Testing allocation strategies... ");
    
    bool success = true;
    
    // Test each allocation strategy
    const atlas_memory_manager::allocation_strategy strategies[] = {
        atlas_memory_manager::ATLAS_ALLOC_FIRST_FIT,
        atlas_memory_manager::ATLAS_ALLOC_BEST_FIT,
        atlas_memory_manager::ATLAS_ALLOC_WORST_FIT
    };
    
    for (auto strategy : strategies) {
        g_memory_manager.allocation_strategy = strategy;
        
        if (!initialize_memory_manager()) {
            success = false;
            break;
        }
        
        atlas_memory_pool* pool = g_memory_manager.pools[ATLAS_MEM_HOST];
        
        // Perform a series of allocations
        std::vector<void*> allocations;
        const size_t sizes[] = {1024, 2048, 512, 4096, 256};
        const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        
        for (size_t i = 0; i < num_sizes; i++) {
            void* ptr = allocate_from_pool(pool, sizes[i], MIN_ALIGNMENT);
            success &= (ptr != nullptr);
            if (ptr) {
                allocations.push_back(ptr);
            }
        }
        
        // Deallocate middle allocation to create hole
        if (allocations.size() >= 3) {
            deallocate_from_pool(pool, allocations[2]);
            
            // Try to allocate something that fits in the hole
            void* new_ptr = allocate_from_pool(pool, 256, MIN_ALIGNMENT);
            success &= (new_ptr != nullptr);
            if (new_ptr) {
                deallocate_from_pool(pool, new_ptr);
            }
        }
        
        // Clean up remaining allocations
        for (void* ptr : allocations) {
            if (ptr != allocations[2]) { // Skip the one we already deallocated
                deallocate_from_pool(pool, ptr);
            }
        }
        
        shutdown_memory_manager();
        
        if (!success) break;
    }
    
    if (success) {
        printf("ok\n");
        return true;
    } else {
        printf("FAILED\n");
        return false;
    }
}

int main(int argc, char** argv) {
    printf("Running ATLAS memory management tests...\n");
    
    int tests_passed = 0;
    int total_tests = 8;
    
    if (test_memory_pool_creation()) tests_passed++;
    if (test_basic_allocation()) tests_passed++;
    if (test_memory_alignment()) tests_passed++;
    if (test_fragmentation_coalescing()) tests_passed++;
    if (test_multithreaded_access()) tests_passed++;
    if (test_pool_statistics()) tests_passed++;
    if (test_memory_types()) tests_passed++;
    if (test_allocation_strategies()) tests_passed++;
    
    printf("\nATLAS memory tests completed: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("All ATLAS memory tests PASSED!\n");
        return 0;
    } else {
        printf("Some ATLAS memory tests FAILED!\n");
        return 1;
    }
}