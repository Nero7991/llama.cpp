#pragma once

// ATLAS Integration for llama-server
// This header provides the integration interface for ATLAS functionality

#include "atlas-server.hpp"
#include "json.hpp"

namespace atlas {

// Simple integration interface that can be called from existing server code
class simple_atlas_integration {
private:
    std::unique_ptr<atlas_server> atlas_srv_;
    bool initialized_;
    
public:
    simple_atlas_integration() : initialized_(false) {
        atlas_server_config config;
        config.enable_memory_persistence = true;
        config.max_concurrent_atlas_requests = 8;
        config.api_latency_budget_ms = 5.0f;
        config.enable_metrics_collection = true;
        
        atlas_srv_ = std::make_unique<atlas_server>(config);
    }
    
    ~simple_atlas_integration() {
        if (atlas_srv_) {
            atlas_srv_->shutdown();
        }
    }
    
    // Initialize ATLAS with model
    bool initialize(llama_model* model) {
        if (!atlas_srv_ || !model) return false;
        
        initialized_ = atlas_srv_->initialize(model);
        if (initialized_) {
            fprintf(stderr, "ATLAS Phase 6A integration initialized successfully\n");
        }
        return initialized_;
    }
    
    // Check if a request has ATLAS parameters
    bool has_atlas_params(const nlohmann::json& request) {
        return request.contains("atlas") && request["atlas"].is_object();
    }
    
    // Process ATLAS-enhanced request
    nlohmann::json process_atlas_request(const nlohmann::json& request) {
        if (!initialized_ || !atlas_srv_) {
            return atlas::error_handling::create_internal_error("ATLAS not initialized");
        }
        
        return atlas_srv_->handle_completion_request(request);
    }
    
    // Get ATLAS metrics
    nlohmann::json get_atlas_metrics() {
        if (!initialized_ || !atlas_srv_) {
            return nlohmann::json{{"error", "ATLAS not initialized"}};
        }
        
        return atlas_srv_->handle_metrics_request();
    }
    
    // Get ATLAS status
    nlohmann::json get_atlas_status() {
        if (!initialized_ || !atlas_srv_) {
            return nlohmann::json{{"status", "disabled"}};
        }
        
        return atlas_srv_->handle_status_request();
    }
    
    // Handle memory operations
    nlohmann::json handle_atlas_memory(const std::string& operation, const nlohmann::json& params) {
        if (!initialized_ || !atlas_srv_) {
            return atlas::error_handling::create_internal_error("ATLAS not initialized");
        }
        
        return atlas_srv_->handle_memory_operation(operation, params);
    }
    
    // Check if ATLAS is healthy
    bool is_atlas_healthy() {
        return initialized_ && atlas_srv_ && atlas_srv_->is_healthy();
    }
};

// Global ATLAS integration instance
extern std::unique_ptr<simple_atlas_integration> g_atlas;

// Convenience functions for existing server code
inline void init_atlas(llama_model* model) {
    if (!g_atlas) {
        g_atlas = std::make_unique<simple_atlas_integration>();
    }
    g_atlas->initialize(model);
}

inline bool is_atlas_request(const nlohmann::json& request) {
    return g_atlas && g_atlas->has_atlas_params(request);
}

inline nlohmann::json process_with_atlas(const nlohmann::json& request) {
    if (!g_atlas) {
        return atlas::error_handling::create_internal_error("ATLAS not available");
    }
    return g_atlas->process_atlas_request(request);
}

inline nlohmann::json get_atlas_info() {
    if (!g_atlas) {
        return nlohmann::json{{"atlas_enabled", false}};
    }
    return nlohmann::json{
        {"atlas_enabled", true},
        {"atlas_healthy", g_atlas->is_atlas_healthy()},
        {"atlas_version", "6A"}
    };
}

// Enhanced response wrapper that adds ATLAS metadata
inline nlohmann::json enhance_response_with_atlas(const nlohmann::json& response, bool was_atlas_processed = false) {
    nlohmann::json enhanced = response;
    
    if (was_atlas_processed) {
        enhanced["atlas"] = {
            {"processed", true},
            {"version", "6A"},
            {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}
        };
    }
    
    return enhanced;
}

} // namespace atlas

// Macro for easy integration in existing server endpoints
#define ATLAS_PROCESS_REQUEST(request_json, response_json) \
    do { \
        if (atlas::is_atlas_request(request_json)) { \
            response_json = atlas::process_with_atlas(request_json); \
            response_json = atlas::enhance_response_with_atlas(response_json, true); \
        } \
    } while(0)

// Macro for adding ATLAS endpoints to httplib server
#define ATLAS_ADD_ENDPOINTS(server) \
    do { \
        server.Get("/v1/atlas/status", [](const httplib::Request& req, httplib::Response& res) { \
            if (atlas::g_atlas) { \
                res.set_content(atlas::g_atlas->get_atlas_status().dump(), "application/json"); \
            } else { \
                res.status = 503; \
                res.set_content("{\"error\":\"ATLAS not available\"}", "application/json"); \
            } \
        }); \
        \
        server.Get("/v1/atlas/metrics", [](const httplib::Request& req, httplib::Response& res) { \
            if (atlas::g_atlas) { \
                res.set_content(atlas::g_atlas->get_atlas_metrics().dump(), "application/json"); \
            } else { \
                res.status = 503; \
                res.set_content("{\"error\":\"ATLAS not available\"}", "application/json"); \
            } \
        }); \
        \
        server.Post("/v1/atlas/memory/([^/]+)", [](const httplib::Request& req, httplib::Response& res) { \
            std::string operation = req.matches[1]; \
            if (atlas::g_atlas) { \
                nlohmann::json params = req.body.empty() ? nlohmann::json{} : nlohmann::json::parse(req.body); \
                res.set_content(atlas::g_atlas->handle_atlas_memory(operation, params).dump(), "application/json"); \
            } else { \
                res.status = 503; \
                res.set_content("{\"error\":\"ATLAS not available\"}", "application/json"); \
            } \
        }); \
    } while(0)