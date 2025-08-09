#pragma once

#include "common.h"
#include "llama.h"
#include "json.hpp"
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <queue>
#include <deque>
#include <condition_variable>

// ATLAS Server Integration for llama-server
namespace atlas {

// Forward declarations
struct llama_context;
struct llama_model;

// ATLAS server configuration
struct atlas_server_config {
    bool enable_memory_persistence = true;
    std::string memory_save_path = "./atlas_memory";
    int memory_auto_save_interval = 300; // seconds
    
    int max_concurrent_atlas_requests = 8;
    float api_latency_budget_ms = 5.0f;
    bool enable_batch_processing = true;
    
    bool enable_metrics_collection = true;
    int metrics_update_interval_ms = 1000;
    int metrics_history_size = 1000;
    
    size_t l1_cache_size = 64 * 1024 * 1024;    // 64MB L1
    size_t l2_cache_size = 256 * 1024 * 1024;   // 256MB L2  
    size_t l3_cache_size = 1024 * 1024 * 1024;  // 1GB L3
};

// ATLAS context wrapper with thread safety
struct atlas_context_wrapper {
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx;
    std::mutex access_mutex;
    std::atomic<uint64_t> request_count{0};
    std::atomic<uint64_t> last_access_time{0};
    std::atomic<bool> is_busy{false};
    
    nlohmann::json atlas_state;
    std::string context_id;
    
    atlas_context_wrapper(llama_context* c) : ctx(c, llama_free) {}
};

// Real-time metrics structure
struct atlas_metrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> active_requests{0}; 
    std::atomic<uint64_t> completed_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<double> max_latency_ms{0.0};
    std::atomic<double> min_latency_ms{999999.0};
    
    std::atomic<uint64_t> memory_usage_bytes{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    
    std::atomic<double> throughput_tokens_per_sec{0.0};
    std::chrono::steady_clock::time_point start_time;
    
    atlas_metrics() : start_time(std::chrono::steady_clock::now()) {}
};

// Thread-safe context pool manager
class atlas_context_pool {
private:
    std::vector<std::unique_ptr<atlas_context_wrapper>> contexts;
    std::mutex pool_mutex;
    std::condition_variable context_available;
    std::atomic<int> available_contexts{0};
    
public:
    void initialize(int pool_size, llama_model* model);
    std::shared_ptr<atlas_context_wrapper> acquire_context(const std::string& session_id = "");
    void release_context(std::shared_ptr<atlas_context_wrapper> ctx);
    void cleanup();
    size_t size() const { return contexts.size(); }
};

// Memory persistence manager
class atlas_memory_manager {
private:
    std::string base_path;
    std::mutex save_mutex;
    std::thread auto_save_thread;
    std::atomic<bool> auto_save_enabled{false};
    
public:
    explicit atlas_memory_manager(const std::string& path);
    ~atlas_memory_manager();
    
    bool save_context_state(const std::string& context_id, const nlohmann::json& state);
    bool load_context_state(const std::string& context_id, nlohmann::json& state);
    bool delete_context_state(const std::string& context_id);
    std::vector<std::string> list_saved_contexts();
    
    void start_auto_save(int interval_seconds);
    void stop_auto_save();
};

// Real-time metrics collector
class atlas_metrics_collector {
private:
    atlas_metrics metrics;
    std::mutex metrics_mutex;
    std::thread collection_thread;
    std::atomic<bool> collection_enabled{false};
    
    std::deque<atlas_metrics> metrics_history;
    size_t max_history_size;
    
public:
    explicit atlas_metrics_collector(size_t history_size = 1000);
    ~atlas_metrics_collector();
    
    void start_collection(int update_interval_ms);
    void stop_collection();
    
    void record_request_start();
    void record_request_end(double latency_ms, bool success);
    void update_memory_usage(uint64_t bytes);
    void record_cache_hit();
    void record_cache_miss();
    void update_throughput(double tokens_per_sec);
    
    atlas_metrics get_current_metrics() const;
    std::vector<atlas_metrics> get_metrics_history(size_t count = 100) const;
    nlohmann::json get_metrics_json() const;
};

// Main ATLAS server integration class
class atlas_server {
private:
    atlas_server_config config;
    std::unique_ptr<atlas_context_pool> context_pool;
    std::unique_ptr<atlas_memory_manager> memory_manager;
    std::unique_ptr<atlas_metrics_collector> metrics_collector;
    
    llama_model* model;
    std::atomic<bool> initialized{false};
    std::mutex server_mutex;
    
public:
    explicit atlas_server(const atlas_server_config& cfg);
    ~atlas_server();
    
    bool initialize(llama_model* model);
    void shutdown();
    
    // Core API methods
    nlohmann::json handle_completion_request(const nlohmann::json& request);
    nlohmann::json handle_atlas_config_request(const nlohmann::json& request);
    nlohmann::json handle_memory_operation(const std::string& operation, const nlohmann::json& params);
    nlohmann::json handle_metrics_request();
    nlohmann::json handle_status_request();
    
    // OpenAI compatibility extensions
    nlohmann::json handle_openai_completion(const nlohmann::json& request);
    nlohmann::json handle_openai_chat_completion(const nlohmann::json& request);
    
    // Configuration
    void update_config(const atlas_server_config& new_config);
    atlas_server_config get_config() const { return config; }
    
    // Health check
    bool is_healthy() const;
    nlohmann::json get_health_status() const;
};

// Request validation utilities
namespace validation {
    bool validate_completion_request(const nlohmann::json& request, std::string& error);
    bool validate_atlas_parameters(const nlohmann::json& params, std::string& error);
    bool validate_memory_operation(const std::string& operation, const nlohmann::json& params, std::string& error);
    nlohmann::json sanitize_request(const nlohmann::json& request);
}

// Error handling utilities
namespace error_handling {
    nlohmann::json create_error_response(int code, const std::string& message, const std::string& type = "api_error");
    nlohmann::json create_validation_error(const std::string& field, const std::string& message);
    nlohmann::json create_rate_limit_error(int retry_after_seconds);
    nlohmann::json create_internal_error(const std::string& message);
}

// Performance utilities
namespace performance {
    class latency_tracker {
    private:
        std::chrono::steady_clock::time_point start_time;
        
    public:
        latency_tracker() : start_time(std::chrono::steady_clock::now()) {}
        
        double elapsed_ms() const {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time);
            return duration.count() / 1000.0;
        }
    };
    
    class request_throttler {
    private:
        std::atomic<int> active_requests{0};
        int max_concurrent;
        
    public:
        explicit request_throttler(int max_concurrent_requests) 
            : max_concurrent(max_concurrent_requests) {}
        
        bool try_acquire() {
            int current = active_requests.load();
            return (current < max_concurrent) && 
                   active_requests.compare_exchange_weak(current, current + 1);
        }
        
        void release() {
            active_requests.fetch_sub(1);
        }
        
        int get_active_count() const { return active_requests.load(); }
    };
}

} // namespace atlas