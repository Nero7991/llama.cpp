#include "atlas-server.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>

namespace atlas {

// ATLAS Context Pool Implementation
void atlas_context_pool::initialize(int pool_size, llama_model* model) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    
    contexts.clear();
    contexts.reserve(pool_size);
    
    for (int i = 0; i < pool_size; ++i) {
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;
        ctx_params.n_threads = std::thread::hardware_concurrency() / pool_size;
        ctx_params.n_threads_batch = ctx_params.n_threads;
        ctx_params.flash_attn = true;
        ctx_params.no_perf = false;
        
        llama_context* ctx = llama_new_context_with_model(model, ctx_params);
        if (ctx) {
            auto wrapper = std::make_unique<atlas_context_wrapper>(ctx);
            wrapper->context_id = "atlas_ctx_" + std::to_string(i);
            contexts.push_back(std::move(wrapper));
        }
    }
    
    available_contexts = contexts.size();
}

std::shared_ptr<atlas_context_wrapper> atlas_context_pool::acquire_context(const std::string& session_id) {
    std::unique_lock<std::mutex> lock(pool_mutex);
    
    if (!context_available.wait_for(lock, std::chrono::milliseconds(100), 
                                   [this]{ return available_contexts.load() > 0; })) {
        return nullptr;
    }
    
    for (auto& ctx : contexts) {
        if (!ctx->is_busy.exchange(true)) {
            ctx->last_access_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            available_contexts--;
            
            return std::shared_ptr<atlas_context_wrapper>(ctx.get(), 
                [this](atlas_context_wrapper* wrapper) {
                    this->release_context(std::shared_ptr<atlas_context_wrapper>(wrapper, [](atlas_context_wrapper*){}));
                });
        }
    }
    
    return nullptr;
}

void atlas_context_pool::release_context(std::shared_ptr<atlas_context_wrapper> ctx) {
    if (!ctx) return;
    
    std::lock_guard<std::mutex> lock(pool_mutex);
    ctx->is_busy = false;
    available_contexts++;
    context_available.notify_one();
}

void atlas_context_pool::cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    contexts.clear();
    available_contexts = 0;
}

// ATLAS Memory Manager Implementation
atlas_memory_manager::atlas_memory_manager(const std::string& path) : base_path(path) {
    std::filesystem::create_directories(base_path);
}

atlas_memory_manager::~atlas_memory_manager() {
    stop_auto_save();
}

bool atlas_memory_manager::save_context_state(const std::string& context_id, const nlohmann::json& state) {
    std::lock_guard<std::mutex> lock(save_mutex);
    
    try {
        std::filesystem::path file_path = std::filesystem::path(base_path) / (context_id + ".json");
        std::ofstream file(file_path);
        if (!file.is_open()) return false;
        
        nlohmann::json full_state = {
            {"context_id", context_id},
            {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"version", "1.0"},
            {"atlas_state", state}
        };
        
        file << full_state.dump(2);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool atlas_memory_manager::load_context_state(const std::string& context_id, nlohmann::json& state) {
    std::lock_guard<std::mutex> lock(save_mutex);
    
    try {
        std::filesystem::path file_path = std::filesystem::path(base_path) / (context_id + ".json");
        if (!std::filesystem::exists(file_path)) return false;
        
        std::ifstream file(file_path);
        if (!file.is_open()) return false;
        
        nlohmann::json full_state;
        file >> full_state;
        
        if (full_state.contains("atlas_state")) {
            state = full_state["atlas_state"];
            return true;
        }
        return false;
    } catch (const std::exception&) {
        return false;
    }
}

bool atlas_memory_manager::delete_context_state(const std::string& context_id) {
    std::lock_guard<std::mutex> lock(save_mutex);
    
    try {
        std::filesystem::path file_path = std::filesystem::path(base_path) / (context_id + ".json");
        return std::filesystem::remove(file_path);
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> atlas_memory_manager::list_saved_contexts() {
    std::lock_guard<std::mutex> lock(save_mutex);
    std::vector<std::string> contexts;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(base_path)) {
            if (entry.path().extension() == ".json") {
                contexts.push_back(entry.path().stem().string());
            }
        }
    } catch (const std::exception&) {
        // Return empty vector on error
    }
    
    return contexts;
}

void atlas_memory_manager::start_auto_save(int interval_seconds) {
    if (auto_save_enabled.load()) return;
    
    auto_save_enabled = true;
    auto_save_thread = std::thread([this, interval_seconds]() {
        while (auto_save_enabled.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));
            // Auto-save logic would be implemented here
            // This is a placeholder for automatic context saving
        }
    });
}

void atlas_memory_manager::stop_auto_save() {
    auto_save_enabled = false;
    if (auto_save_thread.joinable()) {
        auto_save_thread.join();
    }
}

// ATLAS Metrics Collector Implementation
atlas_metrics_collector::atlas_metrics_collector(size_t history_size) : max_history_size(history_size) {}

atlas_metrics_collector::~atlas_metrics_collector() {
    stop_collection();
}

void atlas_metrics_collector::start_collection(int update_interval_ms) {
    if (collection_enabled.load()) return;
    
    collection_enabled = true;
    collection_thread = std::thread([this, update_interval_ms]() {
        while (collection_enabled.load()) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - metrics.start_time);
            
            if (duration.count() > 0) {
                double total_requests = metrics.completed_requests.load();
                if (total_requests > 0) {
                    double throughput = total_requests / duration.count();
                    metrics.throughput_tokens_per_sec = throughput; // Simplified
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                metrics_history.push_back(metrics);
                if (metrics_history.size() > max_history_size) {
                    metrics_history.pop_front();
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(update_interval_ms));
        }
    });
}

void atlas_metrics_collector::stop_collection() {
    collection_enabled = false;
    if (collection_thread.joinable()) {
        collection_thread.join();
    }
}

void atlas_metrics_collector::record_request_start() {
    metrics.total_requests++;
    metrics.active_requests++;
}

void atlas_metrics_collector::record_request_end(double latency_ms, bool success) {
    metrics.active_requests--;
    
    if (success) {
        metrics.completed_requests++;
    } else {
        metrics.failed_requests++;
    }
    
    double current_avg = metrics.avg_latency_ms.load();
    double total_completed = metrics.completed_requests.load();
    if (total_completed > 1) {
        double new_avg = (current_avg * (total_completed - 1) + latency_ms) / total_completed;
        metrics.avg_latency_ms = new_avg;
    } else {
        metrics.avg_latency_ms = latency_ms;
    }
    
    double current_max = metrics.max_latency_ms.load();
    if (latency_ms > current_max) {
        metrics.max_latency_ms = latency_ms;
    }
    
    double current_min = metrics.min_latency_ms.load();
    if (latency_ms < current_min) {
        metrics.min_latency_ms = latency_ms;
    }
}

void atlas_metrics_collector::update_memory_usage(uint64_t bytes) {
    metrics.memory_usage_bytes = bytes;
}

void atlas_metrics_collector::record_cache_hit() {
    metrics.cache_hits++;
}

void atlas_metrics_collector::record_cache_miss() {
    metrics.cache_misses++;
}

void atlas_metrics_collector::update_throughput(double tokens_per_sec) {
    metrics.throughput_tokens_per_sec = tokens_per_sec;
}

atlas_metrics atlas_metrics_collector::get_current_metrics() const {
    return metrics;
}

std::vector<atlas_metrics> atlas_metrics_collector::get_metrics_history(size_t count) const {
    std::lock_guard<std::mutex> lock(metrics_mutex);
    std::vector<atlas_metrics> result;
    
    size_t start_idx = metrics_history.size() > count ? metrics_history.size() - count : 0;
    for (size_t i = start_idx; i < metrics_history.size(); ++i) {
        result.push_back(metrics_history[i]);
    }
    
    return result;
}

nlohmann::json atlas_metrics_collector::get_metrics_json() const {
    auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - metrics.start_time).count();
        
    return {
        {"total_requests", metrics.total_requests.load()},
        {"active_requests", metrics.active_requests.load()},
        {"completed_requests", metrics.completed_requests.load()},
        {"failed_requests", metrics.failed_requests.load()},
        {"avg_latency_ms", metrics.avg_latency_ms.load()},
        {"max_latency_ms", metrics.max_latency_ms.load()},
        {"min_latency_ms", metrics.min_latency_ms.load()},
        {"memory_usage_bytes", metrics.memory_usage_bytes.load()},
        {"cache_hits", metrics.cache_hits.load()},
        {"cache_misses", metrics.cache_misses.load()},
        {"throughput_tokens_per_sec", metrics.throughput_tokens_per_sec.load()},
        {"uptime_seconds", uptime_seconds}
    };
}

// ATLAS Server Main Implementation
atlas_server::atlas_server(const atlas_server_config& cfg) : config(cfg) {
    context_pool = std::make_unique<atlas_context_pool>();
    memory_manager = std::make_unique<atlas_memory_manager>(cfg.memory_save_path);
    metrics_collector = std::make_unique<atlas_metrics_collector>(cfg.metrics_history_size);
}

atlas_server::~atlas_server() {
    shutdown();
}

bool atlas_server::initialize(llama_model* model) {
    std::lock_guard<std::mutex> lock(server_mutex);
    
    if (initialized.load()) return true;
    if (!model) return false;
    
    this->model = model;
    
    context_pool->initialize(config.max_concurrent_atlas_requests, model);
    
    if (config.enable_metrics_collection) {
        metrics_collector->start_collection(config.metrics_update_interval_ms);
    }
    
    if (config.enable_memory_persistence) {
        memory_manager->start_auto_save(config.memory_auto_save_interval);
    }
    
    initialized = true;
    return true;
}

void atlas_server::shutdown() {
    std::lock_guard<std::mutex> lock(server_mutex);
    
    if (!initialized.load()) return;
    
    if (metrics_collector) {
        metrics_collector->stop_collection();
    }
    
    if (memory_manager) {
        memory_manager->stop_auto_save();
    }
    
    if (context_pool) {
        context_pool->cleanup();
    }
    
    initialized = false;
}

nlohmann::json atlas_server::handle_completion_request(const nlohmann::json& request) {
    performance::latency_tracker tracker;
    metrics_collector->record_request_start();
    
    std::string error;
    if (!validation::validate_completion_request(request, error)) {
        metrics_collector->record_request_end(tracker.elapsed_ms(), false);
        return error_handling::create_validation_error("request", error);
    }
    
    auto ctx_wrapper = context_pool->acquire_context();
    if (!ctx_wrapper) {
        metrics_collector->record_request_end(tracker.elapsed_ms(), false);
        return error_handling::create_error_response(503, "No available ATLAS contexts", "resource_exhausted");
    }
    
    try {
        std::lock_guard<std::mutex> ctx_lock(ctx_wrapper->access_mutex);
        
        nlohmann::json response = {
            {"id", "atlas_completion_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())},
            {"object", "text_completion"},
            {"created", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"model", "atlas_enhanced"},
            {"choices", nlohmann::json::array()}
        };
        
        // Actual ATLAS-enhanced inference would happen here
        nlohmann::json choice = {
            {"text", "ATLAS-enhanced completion placeholder - " + request.value("prompt", "")},
            {"index", 0},
            {"logprobs", nullptr},
            {"finish_reason", "stop"}
        };
        response["choices"].push_back(choice);
        
        response["usage"] = {
            {"prompt_tokens", request.value("max_tokens", 100)},
            {"completion_tokens", 50},
            {"total_tokens", request.value("max_tokens", 100) + 50}
        };
        
        response["atlas"] = {
            {"context_id", ctx_wrapper->context_id},
            {"cache_hits", 0},
            {"memory_layers_used", 3},
            {"performance_ms", tracker.elapsed_ms()}
        };
        
        metrics_collector->record_request_end(tracker.elapsed_ms(), true);
        return response;
        
    } catch (const std::exception& e) {
        metrics_collector->record_request_end(tracker.elapsed_ms(), false);
        return error_handling::create_internal_error(std::string("ATLAS processing error: ") + e.what());
    }
}

nlohmann::json atlas_server::handle_atlas_config_request(const nlohmann::json& request) {
    return {
        {"status", "success"},
        {"data", {
            {"current_config", {
                {"max_concurrent_requests", config.max_concurrent_atlas_requests},
                {"api_latency_budget_ms", config.api_latency_budget_ms},
                {"enable_metrics", config.enable_metrics_collection},
                {"enable_persistence", config.enable_memory_persistence}
            }}
        }}
    };
}

nlohmann::json atlas_server::handle_memory_operation(const std::string& operation, const nlohmann::json& params) {
    if (operation == "save") {
        std::string context_id = params.value("context_id", "default");
        nlohmann::json state = params.value("state", nlohmann::json{});
        
        bool success = memory_manager->save_context_state(context_id, state);
        return {
            {"status", success ? "success" : "error"},
            {"operation", "save"},
            {"context_id", context_id},
            {"message", success ? "Context saved successfully" : "Failed to save context"}
        };
    } else if (operation == "load") {
        std::string context_id = params.value("context_id", "default");
        nlohmann::json state;
        
        bool success = memory_manager->load_context_state(context_id, state);
        return {
            {"status", success ? "success" : "error"},
            {"operation", "load"},
            {"context_id", context_id},
            {"data", state},
            {"message", success ? "Context loaded successfully" : "Failed to load context"}
        };
    } else if (operation == "list") {
        auto contexts = memory_manager->list_saved_contexts();
        return {
            {"status", "success"},
            {"operation", "list"},
            {"contexts", contexts}
        };
    } else if (operation == "delete") {
        std::string context_id = params.value("context_id", "");
        if (context_id.empty()) {
            return error_handling::create_validation_error("context_id", "Context ID is required for delete operation");
        }
        
        bool success = memory_manager->delete_context_state(context_id);
        return {
            {"status", success ? "success" : "error"},
            {"operation", "delete"},
            {"context_id", context_id},
            {"message", success ? "Context deleted successfully" : "Failed to delete context"}
        };
    }
    
    return error_handling::create_validation_error("operation", "Invalid operation: " + operation);
}

nlohmann::json atlas_server::handle_metrics_request() {
    return {
        {"status", "success"},
        {"data", {
            {"current", metrics_collector->get_metrics_json()},
            {"history_size", metrics_collector->get_metrics_history().size()},
            {"server_health", is_healthy()}
        }}
    };
}

nlohmann::json atlas_server::handle_status_request() {
    return {
        {"status", "success"},
        {"data", get_health_status()}
    };
}

nlohmann::json atlas_server::handle_openai_completion(const nlohmann::json& request) {
    // Convert OpenAI format to ATLAS format and delegate
    return handle_completion_request(request);
}

nlohmann::json atlas_server::handle_openai_chat_completion(const nlohmann::json& request) {
    // Convert chat format to completion format
    std::string combined_prompt;
    if (request.contains("messages") && request["messages"].is_array()) {
        for (const auto& msg : request["messages"]) {
            if (msg.contains("content")) {
                combined_prompt += msg["content"].get<std::string>() + "\n";
            }
        }
    }
    
    nlohmann::json completion_req = request;
    completion_req["prompt"] = combined_prompt;
    
    auto response = handle_completion_request(completion_req);
    
    // Convert back to chat format
    if (response.contains("choices")) {
        for (auto& choice : response["choices"]) {
            if (choice.contains("text")) {
                choice["message"] = {
                    {"role", "assistant"},
                    {"content", choice["text"]}
                };
                choice.erase("text");
            }
        }
    }
    
    return response;
}

void atlas_server::update_config(const atlas_server_config& new_config) {
    std::lock_guard<std::mutex> lock(server_mutex);
    config = new_config;
}

bool atlas_server::is_healthy() const {
    return initialized.load() && 
           context_pool && 
           context_pool->size() > 0 &&
           metrics_collector &&
           memory_manager;
}

nlohmann::json atlas_server::get_health_status() const {
    return {
        {"healthy", is_healthy()},
        {"initialized", initialized.load()},
        {"context_pool_size", context_pool ? context_pool->size() : 0},
        {"metrics_enabled", config.enable_metrics_collection},
        {"memory_persistence", config.enable_memory_persistence},
        {"uptime_seconds", std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - metrics_collector->get_current_metrics().start_time).count()}
    };
}

} // namespace atlas