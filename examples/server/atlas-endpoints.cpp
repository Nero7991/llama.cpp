#include "atlas-server.hpp"
#ifdef SERVER_VERBOSE
#include "httplib.h"
#endif

namespace atlas {

// Request validation implementations
namespace validation {
    bool validate_completion_request(const nlohmann::json& request, std::string& error) {
        if (!request.contains("prompt")) {
            error = "Missing required field: prompt";
            return false;
        }
        
        if (!request["prompt"].is_string()) {
            error = "Field 'prompt' must be a string";
            return false;
        }
        
        if (request.contains("max_tokens") && !request["max_tokens"].is_number_integer()) {
            error = "Field 'max_tokens' must be an integer";
            return false;
        }
        
        if (request.contains("temperature")) {
            if (!request["temperature"].is_number()) {
                error = "Field 'temperature' must be a number";
                return false;
            }
            double temp = request["temperature"].get<double>();
            if (temp < 0.0 || temp > 2.0) {
                error = "Field 'temperature' must be between 0.0 and 2.0";
                return false;
            }
        }
        
        if (request.contains("atlas") && request["atlas"].is_object()) {
            return validate_atlas_parameters(request["atlas"], error);
        }
        
        return true;
    }
    
    bool validate_atlas_parameters(const nlohmann::json& params, std::string& error) {
        if (params.contains("memory_layers")) {
            if (!params["memory_layers"].is_number_integer()) {
                error = "atlas.memory_layers must be an integer";
                return false;
            }
            int layers = params["memory_layers"].get<int>();
            if (layers < 1 || layers > 10) {
                error = "atlas.memory_layers must be between 1 and 10";
                return false;
            }
        }
        
        if (params.contains("cache_strategy")) {
            if (!params["cache_strategy"].is_string()) {
                error = "atlas.cache_strategy must be a string";
                return false;
            }
            const auto& strategy = params["cache_strategy"].get<std::string>();
            if (strategy != "lru" && strategy != "lfu" && strategy != "adaptive") {
                error = "atlas.cache_strategy must be one of: lru, lfu, adaptive";
                return false;
            }
        }
        
        if (params.contains("session_id")) {
            if (!params["session_id"].is_string()) {
                error = "atlas.session_id must be a string";
                return false;
            }
            const auto& session_id = params["session_id"].get<std::string>();
            if (session_id.length() > 128) {
                error = "atlas.session_id must be 128 characters or less";
                return false;
            }
        }
        
        if (params.contains("batch_size")) {
            if (!params["batch_size"].is_number_integer()) {
                error = "atlas.batch_size must be an integer";
                return false;
            }
            int batch_size = params["batch_size"].get<int>();
            if (batch_size < 1 || batch_size > 32) {
                error = "atlas.batch_size must be between 1 and 32";
                return false;
            }
        }
        
        return true;
    }
    
    bool validate_memory_operation(const std::string& operation, const nlohmann::json& params, std::string& error) {
        if (operation != "save" && operation != "load" && operation != "delete" && operation != "list") {
            error = "Invalid operation. Must be one of: save, load, delete, list";
            return false;
        }
        
        if (operation == "save" || operation == "load" || operation == "delete") {
            if (!params.contains("context_id") || !params["context_id"].is_string()) {
                error = "context_id is required and must be a string for " + operation + " operation";
                return false;
            }
            
            const auto& context_id = params["context_id"].get<std::string>();
            if (context_id.empty() || context_id.length() > 64) {
                error = "context_id must be non-empty and 64 characters or less";
                return false;
            }
            
            // Validate context_id contains only safe characters
            if (context_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos) {
                error = "context_id can only contain letters, numbers, underscores, and hyphens";
                return false;
            }
        }
        
        return true;
    }
    
    nlohmann::json sanitize_request(const nlohmann::json& request) {
        nlohmann::json sanitized = request;
        
        // Sanitize prompt length
        if (sanitized.contains("prompt") && sanitized["prompt"].is_string()) {
            std::string prompt = sanitized["prompt"].get<std::string>();
            if (prompt.length() > 100000) { // 100KB limit
                prompt = prompt.substr(0, 100000);
                sanitized["prompt"] = prompt;
            }
        }
        
        // Sanitize max_tokens
        if (sanitized.contains("max_tokens") && sanitized["max_tokens"].is_number_integer()) {
            int max_tokens = sanitized["max_tokens"].get<int>();
            if (max_tokens > 4096) {
                sanitized["max_tokens"] = 4096;
            } else if (max_tokens < 1) {
                sanitized["max_tokens"] = 1;
            }
        }
        
        // Sanitize temperature
        if (sanitized.contains("temperature") && sanitized["temperature"].is_number()) {
            double temperature = sanitized["temperature"].get<double>();
            if (temperature > 2.0) {
                sanitized["temperature"] = 2.0;
            } else if (temperature < 0.0) {
                sanitized["temperature"] = 0.0;
            }
        }
        
        return sanitized;
    }
}

// Error handling implementations
namespace error_handling {
    nlohmann::json create_error_response(int code, const std::string& message, const std::string& type) {
        return {
            {"error", {
                {"code", code},
                {"message", message},
                {"type", type},
                {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            }}
        };
    }
    
    nlohmann::json create_validation_error(const std::string& field, const std::string& message) {
        return create_error_response(400, "Validation error in field '" + field + "': " + message, "validation_error");
    }
    
    nlohmann::json create_rate_limit_error(int retry_after_seconds) {
        return {
            {"error", {
                {"code", 429},
                {"message", "Too many requests. Please try again later."},
                {"type", "rate_limit_exceeded"},
                {"retry_after", retry_after_seconds},
                {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            }}
        };
    }
    
    nlohmann::json create_internal_error(const std::string& message) {
        return create_error_response(500, "Internal server error: " + message, "internal_error");
    }
}

#ifdef SERVER_VERBOSE
// HTTP endpoint integration class
class atlas_endpoint_handler {
private:
    std::unique_ptr<atlas_server> atlas_srv;
    performance::request_throttler throttler;
    
public:
    atlas_endpoint_handler(const atlas_server_config& config) 
        : atlas_srv(std::make_unique<atlas_server>(config)),
          throttler(config.max_concurrent_atlas_requests) {}
    
    bool initialize(llama_model* model) {
        return atlas_srv->initialize(model);
    }
    
    void register_endpoints(httplib::Server& server) {
        // ATLAS-enhanced completions endpoint
        server.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle_completion_endpoint(req, res);
        });
        
        // OpenAI-compatible chat completions with ATLAS extensions
        server.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completion_endpoint(req, res);
        });
        
        // ATLAS configuration endpoints
        server.Post("/v1/atlas/config", [this](const httplib::Request& req, httplib::Response& res) {
            handle_config_endpoint(req, res, true);
        });
        
        server.Get("/v1/atlas/config", [this](const httplib::Request& req, httplib::Response& res) {
            handle_config_endpoint(req, res, false);
        });
        
        // Memory management endpoints
        server.Post("/v1/atlas/memory/([^/]+)", [this](const httplib::Request& req, httplib::Response& res) {
            std::string operation = req.matches[1];
            handle_memory_endpoint(req, res, operation);
        });
        
        // Real-time metrics endpoint
        server.Get("/v1/atlas/metrics", [this](const httplib::Request& req, httplib::Response& res) {
            handle_metrics_endpoint(req, res);
        });
        
        // Health check endpoint
        server.Get("/v1/atlas/health", [this](const httplib::Request& req, httplib::Response& res) {
            handle_health_endpoint(req, res);
        });
        
        // Server-sent events for real-time metrics
        server.Get("/v1/atlas/metrics/stream", [this](const httplib::Request& req, httplib::Response& res) {
            handle_metrics_stream_endpoint(req, res);
        });
    }
    
    void shutdown() {
        if (atlas_srv) {
            atlas_srv->shutdown();
        }
    }

private:
    void handle_completion_endpoint(const httplib::Request& req, httplib::Response& res) {
        if (!throttler.try_acquire()) {
            res.status = 429;
            res.set_content(error_handling::create_rate_limit_error(1).dump(), "application/json");
            return;
        }
        
        auto release_guard = [this]() { throttler.release(); };
        std::unique_ptr<void, decltype(release_guard)> guard(nullptr, release_guard);
        
        try {
            nlohmann::json request = nlohmann::json::parse(req.body);
            request = validation::sanitize_request(request);
            nlohmann::json response = atlas_srv->handle_completion_request(request);
            
            res.set_content(response.dump(), "application/json");
            res.set_header("X-ATLAS-Enabled", "true");
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("X-ATLAS-Version", "6A");
            
        } catch (const nlohmann::json::parse_error& e) {
            res.status = 400;
            res.set_content(error_handling::create_validation_error("body", 
                "Invalid JSON: " + std::string(e.what())).dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_chat_completion_endpoint(const httplib::Request& req, httplib::Response& res) {
        if (!throttler.try_acquire()) {
            res.status = 429;
            res.set_content(error_handling::create_rate_limit_error(1).dump(), "application/json");
            return;
        }
        
        auto release_guard = [this]() { throttler.release(); };
        std::unique_ptr<void, decltype(release_guard)> guard(nullptr, release_guard);
        
        try {
            nlohmann::json request = nlohmann::json::parse(req.body);
            nlohmann::json response = atlas_srv->handle_openai_chat_completion(request);
            
            res.set_content(response.dump(), "application/json");
            res.set_header("X-ATLAS-Enabled", "true");
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("X-ATLAS-Version", "6A");
            
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_config_endpoint(const httplib::Request& req, httplib::Response& res, bool is_post) {
        try {
            nlohmann::json response;
            if (is_post) {
                nlohmann::json request = req.body.empty() ? nlohmann::json{} : nlohmann::json::parse(req.body);
                response = atlas_srv->handle_atlas_config_request(request);
            } else {
                response = {
                    {"status", "success"},
                    {"data", {
                        {"config", {
                            {"version", "6A"},
                            {"enabled", true},
                            {"features", {"memory_layers", "cache_strategy", "session_persistence"}},
                            {"limits", {
                                {"max_concurrent_requests", 32},
                                {"max_prompt_length", 100000},
                                {"max_tokens", 4096}
                            }}
                        }},
                        {"health", atlas_srv->get_health_status()}
                    }}
                };
            }
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_memory_endpoint(const httplib::Request& req, httplib::Response& res, const std::string& operation) {
        try {
            nlohmann::json params = req.body.empty() ? nlohmann::json{} : nlohmann::json::parse(req.body);
            
            std::string error;
            if (!validation::validate_memory_operation(operation, params, error)) {
                res.status = 400;
                res.set_content(error_handling::create_validation_error("operation", error).dump(), "application/json");
                return;
            }
            
            nlohmann::json response = atlas_srv->handle_memory_operation(operation, params);
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_metrics_endpoint(const httplib::Request& req, httplib::Response& res) {
        try {
            nlohmann::json response = atlas_srv->handle_metrics_request();
            res.set_content(response.dump(), "application/json");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-ATLAS-Metrics-Version", "1.0");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_health_endpoint(const httplib::Request& req, httplib::Response& res) {
        try {
            bool healthy = atlas_srv->is_healthy();
            nlohmann::json response = {
                {"status", healthy ? "healthy" : "unhealthy"},
                {"data", atlas_srv->get_health_status()},
                {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            };
            
            if (!healthy) {
                res.status = 503;
            }
            
            res.set_content(response.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_handling::create_internal_error(e.what()).dump(), "application/json");
        }
    }
    
    void handle_metrics_stream_endpoint(const httplib::Request& req, httplib::Response& res) {
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");
        
        // Stream metrics updates for 60 seconds
        for (int i = 0; i < 60; ++i) {
            try {
                nlohmann::json metrics = atlas_srv->handle_metrics_request();
                std::string event_data = "data: " + metrics.dump() + "\n\n";
                res.set_content(event_data, "text/event-stream");
                
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } catch (const std::exception&) {
                break;
            }
        }
    }
};

// Global ATLAS integration instance
static std::unique_ptr<atlas_endpoint_handler> g_atlas_handler;

// Integration function to be called from server.cpp
void integrate_atlas_with_server(httplib::Server& server, llama_model* model, const atlas_server_config& config) {
    g_atlas_handler = std::make_unique<atlas_endpoint_handler>(config);
    
    if (g_atlas_handler->initialize(model)) {
        g_atlas_handler->register_endpoints(server);
        
        // Add middleware for ATLAS request logging
        server.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
            if (req.path.find("/v1/atlas") == 0) {
                res.set_header("X-ATLAS-Version", "6A");
                res.set_header("X-ATLAS-Server", "llama-server");
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        
        // Error handler for ATLAS endpoints
        server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
            if (req.path.find("/v1/atlas") == 0) {
                nlohmann::json error_response = error_handling::create_internal_error("Internal server error");
                res.set_content(error_response.dump(), "application/json");
            }
        });
        
        // Graceful shutdown handler
        std::atexit([]() {
            if (g_atlas_handler) {
                g_atlas_handler->shutdown();
            }
        });
        
        fprintf(stderr, "ATLAS Phase 6A integration initialized successfully\n");
    } else {
        fprintf(stderr, "Failed to initialize ATLAS integration\n");
    }
}

// Cleanup function
void cleanup_atlas_integration() {
    if (g_atlas_handler) {
        g_atlas_handler->shutdown();
        g_atlas_handler.reset();
    }
}

#endif // SERVER_VERBOSE

} // namespace atlas