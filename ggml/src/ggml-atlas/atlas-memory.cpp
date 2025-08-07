#include "../../include/atlas/atlas-types.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <pthread.h>

// Memory block header for tracking allocations
typedef struct atlas_memory_block {
    size_t size;
    size_t alignment;
    bool is_free;
    struct atlas_memory_block* next;
    struct atlas_memory_block* prev;
} atlas_memory_block_t;

// Internal memory pool structure
typedef struct atlas_memory_pool_internal {
    atlas_memory_pool_t base;
    atlas_memory_block_t* blocks;
    pthread_mutex_t mutex;
    size_t num_allocations;
    size_t peak_usage;
} atlas_memory_pool_internal_t;

// Helper function to align size
static size_t align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// Helper function to align pointer
static void* align_ptr(void* ptr, size_t alignment) {
    uintptr_t addr = (uintptr_t)ptr;
    uintptr_t aligned = (addr + alignment - 1) & ~(alignment - 1);
    return (void*)aligned;
}

// Create a new memory pool
atlas_memory_pool_t* atlas_memory_pool_create(size_t initial_size, atlas_memory_pool_flags_t flags) {
    if (initial_size == 0) {
        initial_size = ATLAS_DEFAULT_MEMORY_POOL_SIZE;
    }
    
    // Allocate internal structure
    atlas_memory_pool_internal_t* pool = (atlas_memory_pool_internal_t*)calloc(1, sizeof(atlas_memory_pool_internal_t));
    if (!pool) {
        return NULL;
    }
    
    // Initialize base structure
    pool->base.total_size = align_size(initial_size, ATLAS_DEFAULT_ALIGNMENT);
    pool->base.used_size = 0;
    pool->base.alignment = ATLAS_DEFAULT_ALIGNMENT;
    pool->base.flags = flags;
    pool->base.allocation_strategy = ATLAS_ALLOC_FIRST_FIT;
    
    // Allocate memory buffer
    pool->base.base_ptr = aligned_alloc(pool->base.alignment, pool->base.total_size);
    if (!pool->base.base_ptr) {
        free(pool);
        return NULL;
    }
    
    // Zero-initialize if requested
    if (flags & ATLAS_MEMORY_POOL_FLAG_ZERO_INIT) {
        memset(pool->base.base_ptr, 0, pool->base.total_size);
    }
    
    // Create initial free block
    pool->blocks = (atlas_memory_block_t*)calloc(1, sizeof(atlas_memory_block_t));
    if (!pool->blocks) {
        free(pool->base.base_ptr);
        free(pool);
        return NULL;
    }
    
    pool->blocks->size = pool->base.total_size;
    pool->blocks->alignment = pool->base.alignment;
    pool->blocks->is_free = true;
    pool->blocks->next = NULL;
    pool->blocks->prev = NULL;
    
    // Initialize mutex if thread-safe
    if (flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_init(&pool->mutex, NULL);
    }
    
    pool->num_allocations = 0;
    pool->peak_usage = 0;
    
    return &pool->base;
}

// Destroy a memory pool
void atlas_memory_pool_destroy(atlas_memory_pool_t* pool) {
    if (!pool) {
        return;
    }
    
    atlas_memory_pool_internal_t* internal = (atlas_memory_pool_internal_t*)pool;
    
    // Destroy mutex if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_destroy(&internal->mutex);
    }
    
    // Free block list
    atlas_memory_block_t* block = internal->blocks;
    while (block) {
        atlas_memory_block_t* next = block->next;
        free(block);
        block = next;
    }
    
    // Free memory buffer
    free(pool->base_ptr);
    
    // Free pool structure
    free(internal);
}

// Reset pool to initial state
atlas_status_t atlas_memory_pool_reset(atlas_memory_pool_t* pool) {
    if (!pool) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    atlas_memory_pool_internal_t* internal = (atlas_memory_pool_internal_t*)pool;
    
    // Lock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_lock(&internal->mutex);
    }
    
    // Free all blocks except the first
    atlas_memory_block_t* block = internal->blocks->next;
    while (block) {
        atlas_memory_block_t* next = block->next;
        free(block);
        block = next;
    }
    
    // Reset first block
    internal->blocks->size = pool->total_size;
    internal->blocks->is_free = true;
    internal->blocks->next = NULL;
    internal->blocks->prev = NULL;
    
    pool->used_size = 0;
    internal->num_allocations = 0;
    
    // Zero memory if requested
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_ZERO_INIT) {
        memset(pool->base_ptr, 0, pool->total_size);
    }
    
    // Unlock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_unlock(&internal->mutex);
    }
    
    return ATLAS_STATUS_SUCCESS;
}

// Find a suitable free block based on allocation strategy
static atlas_memory_block_t* find_free_block(atlas_memory_pool_internal_t* pool, 
                                            size_t size, size_t alignment) {
    atlas_memory_block_t* best = NULL;
    size_t best_size = SIZE_MAX;
    
    atlas_memory_block_t* block = pool->blocks;
    
    switch (pool->base.allocation_strategy) {
        case ATLAS_ALLOC_FIRST_FIT:
            // Return first block that fits
            while (block) {
                if (block->is_free && block->size >= size) {
                    return block;
                }
                block = block->next;
            }
            break;
            
        case ATLAS_ALLOC_BEST_FIT:
            // Find smallest block that fits
            while (block) {
                if (block->is_free && block->size >= size && block->size < best_size) {
                    best = block;
                    best_size = block->size;
                }
                block = block->next;
            }
            return best;
            
        case ATLAS_ALLOC_WORST_FIT:
            // Find largest block
            while (block) {
                if (block->is_free && block->size >= size && block->size > best_size) {
                    best = block;
                    best_size = block->size;
                }
                block = block->next;
            }
            return best;
    }
    
    return NULL;
}

// Allocate memory from pool
atlas_status_t atlas_memory_alloc(atlas_memory_pool_t* pool, size_t size, 
                                 size_t alignment, void** ptr) {
    if (!pool || !ptr || size == 0) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    atlas_memory_pool_internal_t* internal = (atlas_memory_pool_internal_t*)pool;
    
    // Use pool alignment if not specified
    if (alignment == 0) {
        alignment = pool->alignment;
    }
    
    // Align size
    size = align_size(size, alignment);
    
    // Lock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_lock(&internal->mutex);
    }
    
    // Find suitable block
    atlas_memory_block_t* block = find_free_block(internal, size, alignment);
    
    if (!block) {
        // Check if pool can grow
        if (pool->flags & ATLAS_MEMORY_POOL_FLAG_GROWABLE) {
            // Pool growing not implemented in this basic version
            if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
                pthread_mutex_unlock(&internal->mutex);
            }
            return ATLAS_STATUS_OUT_OF_MEMORY;
        }
        
        if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
            pthread_mutex_unlock(&internal->mutex);
        }
        return ATLAS_STATUS_OUT_OF_MEMORY;
    }
    
    // Split block if necessary
    if (block->size > size + sizeof(atlas_memory_block_t)) {
        atlas_memory_block_t* new_block = (atlas_memory_block_t*)calloc(1, sizeof(atlas_memory_block_t));
        if (new_block) {
            new_block->size = block->size - size;
            new_block->alignment = alignment;
            new_block->is_free = true;
            new_block->next = block->next;
            new_block->prev = block;
            
            if (block->next) {
                block->next->prev = new_block;
            }
            
            block->next = new_block;
            block->size = size;
        }
    }
    
    // Mark block as used
    block->is_free = false;
    block->alignment = alignment;
    
    // Calculate pointer offset from pool base
    size_t offset = 0;
    atlas_memory_block_t* curr = internal->blocks;
    while (curr != block) {
        offset += curr->size;
        curr = curr->next;
    }
    
    *ptr = (char*)pool->base_ptr + offset;
    
    // Update statistics
    pool->used_size += size;
    internal->num_allocations++;
    
    if (pool->used_size > internal->peak_usage) {
        internal->peak_usage = pool->used_size;
    }
    
    // Unlock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_unlock(&internal->mutex);
    }
    
    return ATLAS_STATUS_SUCCESS;
}

// Free memory back to pool
atlas_status_t atlas_memory_free(atlas_memory_pool_t* pool, void* ptr) {
    if (!pool || !ptr) {
        return ATLAS_STATUS_INVALID_ARGUMENT;
    }
    
    atlas_memory_pool_internal_t* internal = (atlas_memory_pool_internal_t*)pool;
    
    // Lock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_lock(&internal->mutex);
    }
    
    // Find block corresponding to pointer
    size_t offset = (char*)ptr - (char*)pool->base_ptr;
    size_t current_offset = 0;
    atlas_memory_block_t* block = internal->blocks;
    
    while (block) {
        if (current_offset == offset && !block->is_free) {
            // Found the block
            block->is_free = true;
            pool->used_size -= block->size;
            internal->num_allocations--;
            
            // Coalesce with previous block if free
            if (block->prev && block->prev->is_free) {
                block->prev->size += block->size;
                block->prev->next = block->next;
                if (block->next) {
                    block->next->prev = block->prev;
                }
                atlas_memory_block_t* to_free = block;
                block = block->prev;
                free(to_free);
            }
            
            // Coalesce with next block if free
            if (block->next && block->next->is_free) {
                block->size += block->next->size;
                atlas_memory_block_t* to_free = block->next;
                block->next = to_free->next;
                if (to_free->next) {
                    to_free->next->prev = block;
                }
                free(to_free);
            }
            
            // Unlock if thread-safe
            if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
                pthread_mutex_unlock(&internal->mutex);
            }
            
            return ATLAS_STATUS_SUCCESS;
        }
        
        current_offset += block->size;
        block = block->next;
    }
    
    // Unlock if thread-safe
    if (pool->flags & ATLAS_MEMORY_POOL_FLAG_THREAD_SAFE) {
        pthread_mutex_unlock(&internal->mutex);
    }
    
    return ATLAS_STATUS_INVALID_ARGUMENT;
}

// Get used memory size
size_t atlas_memory_pool_get_used(const atlas_memory_pool_t* pool) {
    if (!pool) {
        return 0;
    }
    return pool->used_size;
}

// Get available memory size
size_t atlas_memory_pool_get_available(const atlas_memory_pool_t* pool) {
    if (!pool) {
        return 0;
    }
    return pool->total_size - pool->used_size;
}

// Get number of allocations
size_t atlas_memory_pool_get_num_allocations(const atlas_memory_pool_t* pool) {
    if (!pool) {
        return 0;
    }
    const atlas_memory_pool_internal_t* internal = (const atlas_memory_pool_internal_t*)pool;
    return internal->num_allocations;
}

// Get peak usage
size_t atlas_memory_pool_get_peak_usage(const atlas_memory_pool_t* pool) {
    if (!pool) {
        return 0;
    }
    const atlas_memory_pool_internal_t* internal = (const atlas_memory_pool_internal_t*)pool;
    return internal->peak_usage;
}