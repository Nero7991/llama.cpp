## Summary

Implement llama-server integration for ATLAS with comprehensive API support, enabling HTTP/REST endpoints for ATLAS-enhanced inference. This issue focuses on server infrastructure, API design, and integration with existing llama-server architecture.

## Background

llama-server provides HTTP API access to llama.cpp inference. ATLAS integration requires extending this server to support:
- ATLAS configuration via API parameters
- Real-time memory state monitoring
- ATLAS-specific endpoints for advanced features
- Seamless integration with existing completion endpoints

## Implementation Requirements

### 1. Server Infrastructure Extensions

#### ATLAS Server Context
```c
// Extended server context for ATLAS
struct llama_server_context {
    // ... existing fields ...
    
#ifdef GGML_USE_ATLAS
    bool atlas_enabled;
    struct atlas_context atlas_ctx;
    struct atlas_server_config atlas_config;
    
    // Memory persistence
    char* atlas_memory_file;
    bool auto_save_memory;
    bool auto_load_memory;
    
    // API statistics
    struct atlas_api_stats {
        int requests_processed;
        float avg_memory_utilization;
        float avg_window_fill_ratio;
        int muon_iterations_total;
        double total_omega_loss;
    } atlas_stats;
#endif
};

// ATLAS server configuration
struct atlas_server_config {
    bool enabled;
    int window_size;
    float blend_ratio;
    float learning_rate;
    int polynomial_degree;
    bool use_muon;
    int memory_dim;
    int update_frequency;
    int warmup_tokens;
    
    // Server-specific options
    bool enable_memory_endpoints;
    bool enable_realtime_stats;
    int stats_update_interval;
    bool log_atlas_operations;
};
```

### 2. API Endpoint Extensions

#### Enhanced Completion Endpoint
```json
POST /v1/completions
{
  "prompt": "Your text here...",
  "max_tokens": 2048,
  "temperature": 0.7,
  
  // ATLAS-specific parameters
  "atlas": {
    "enabled": true,
    "window_size": 512,
    "blend_ratio": 0.5,
    "learning_rate": 0.001,
    "polynomial_degree": 3,
    "use_muon": true,
    "warmup_tokens": 128
  }
}

// Response includes ATLAS statistics
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "choices": [
    {
      "text": "Generated text...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "atlas_stats": {
    "memory_utilization": 0.87,
    "window_fill_ratio": 0.94,
    "omega_loss": 0.023,
    "muon_iterations": 4,
    "blend_ratio_actual": 0.52
  }
}
```

#### ATLAS Management Endpoints
```c
// New ATLAS-specific endpoints
GET /v1/atlas/status          // Get current ATLAS status
GET /v1/atlas/memory          // Get memory state information
POST /v1/atlas/memory/save    // Save memory to file
POST /v1/atlas/memory/load    // Load memory from file
POST /v1/atlas/memory/reset   // Reset memory state
GET /v1/atlas/stats           // Detailed statistics
POST /v1/atlas/config         // Update ATLAS configuration
```

### 3. Server Implementation

#### ATLAS Configuration Loading
```c
// Load ATLAS configuration from command line and config file
bool llama_server_load_atlas_config(
    struct llama_server_context* ctx,
    int argc, char** argv,
    const char* config_file_path
) {
#ifdef GGML_USE_ATLAS
    // Parse command line arguments
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--atlas") == 0) {
            ctx->atlas_enabled = true;
        } else if (strcmp(argv[i], "--atlas-window") == 0) {
            ctx->atlas_config.window_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--atlas-blend") == 0) {
            ctx->atlas_config.blend_ratio = atof(argv[++i]);
        }
        // ... more parameters
    }
    
    // Load from JSON config file if provided
    if (config_file_path) {
        load_atlas_config_from_json(ctx, config_file_path);
    }
    
    return true;
#else
    return false;
#endif
}

// Initialize ATLAS server components
bool llama_server_init_atlas(struct llama_server_context* ctx) {
    if (!ctx->atlas_enabled) return true;
    
    // Initialize ATLAS context
    int result = llama_atlas_init(&ctx->llama_ctx, &ctx->atlas_config);
    if (result != 0) {
        LOG_ERROR("Failed to initialize ATLAS: %d\n", result);
        return false;
    }
    
    // Set up auto-load memory if file exists
    if (ctx->atlas_config.auto_load_memory) {
        char default_memory_file[256];
        snprintf(default_memory_file, sizeof(default_memory_file), 
                "%s_memory.atlas", ctx->model_filename);
        
        if (file_exists(default_memory_file)) {
            llama_atlas_load_memory(&ctx->atlas_ctx, default_memory_file);
            LOG_INFO("Loaded ATLAS memory from %s\n", default_memory_file);
        }
    }
    
    return true;
}
```

#### Request Processing with ATLAS
```c
// Process completion request with ATLAS support
json process_completion_request(
    struct llama_server_context* ctx,
    const json& request
) {
    // Parse standard completion parameters
    std::string prompt = request["prompt"];
    int max_tokens = request.value("max_tokens", 100);
    
    // Parse ATLAS parameters if present
    bool use_atlas = ctx->atlas_enabled;
    if (request.contains("atlas")) {
        const auto& atlas_config = request["atlas"];
        
        // Override ATLAS settings for this request
        if (atlas_config.contains("enabled")) {
            use_atlas = atlas_config["enabled"];
        }
        
        if (use_atlas && atlas_config.contains("window_size")) {
            llama_atlas_set_window_size(&ctx->atlas_ctx, 
                                       atlas_config["window_size"]);
        }
        
        // ... other parameter overrides
    }
    
    // Generate response
    std::string response_text;
    struct atlas_inference_stats stats = {0};
    
    if (use_atlas) {
        response_text = generate_with_atlas(ctx, prompt, max_tokens, &stats);
    } else {
        response_text = generate_standard(ctx, prompt, max_tokens);
    }
    
    // Prepare response
    json response = {
        {"id", generate_completion_id()},
        {"object", "text_completion"},
        {"choices", json::array({
            {
                {"text", response_text},
                {"index", 0},
                {"finish_reason", "length"}
            }
        })}
    };
    
    // Add ATLAS statistics if enabled
    if (use_atlas) {
        response["atlas_stats"] = {
            {"memory_utilization", stats.memory_utilization},
            {"window_fill_ratio", stats.window_fill_ratio},
            {"omega_loss", stats.omega_loss},
            {"muon_iterations", stats.muon_iterations},
            {"blend_ratio_actual", stats.actual_blend_ratio}
        };
    }
    
    return response;
}
```

### 4. ATLAS Management Endpoints

#### Memory State Management
```c
// GET /v1/atlas/memory
json handle_atlas_memory_get(struct llama_server_context* ctx) {
    if (!ctx->atlas_enabled) {
        return json{{"error", "ATLAS not enabled"}};
    }
    
    struct atlas_memory_info info;
    llama_atlas_get_memory_info(&ctx->atlas_ctx, &info);
    
    return json{
        {"window_size", info.window_size},
        {"window_fill", info.current_fill},
        {"memory_capacity", info.memory_capacity},
        {"memory_used", info.memory_used},
        {"total_updates", info.total_updates},
        {"last_update_time", info.last_update_timestamp}
    };
}

// POST /v1/atlas/memory/save
json handle_atlas_memory_save(struct llama_server_context* ctx, 
                              const json& request) {
    if (!ctx->atlas_enabled) {
        return json{{"error", "ATLAS not enabled"}};
    }
    
    std::string filename = request.value("filename", "");
    if (filename.empty()) {
        // Generate default filename
        filename = std::string(ctx->model_filename) + "_memory.atlas";
    }
    
    int result = llama_atlas_save_memory(&ctx->atlas_ctx, filename.c_str());
    
    if (result == 0) {
        return json{
            {"success", true},
            {"filename", filename},
            {"timestamp", get_current_timestamp()}
        };
    } else {
        return json{
            {"success", false},
            {"error", "Failed to save memory"},
            {"error_code", result}
        };
    }
}

// POST /v1/atlas/memory/load
json handle_atlas_memory_load(struct llama_server_context* ctx,
                              const json& request) {
    if (!ctx->atlas_enabled) {
        return json{{"error", "ATLAS not enabled"}};
    }
    
    std::string filename = request["filename"];
    int result = llama_atlas_load_memory(&ctx->atlas_ctx, filename.c_str());
    
    if (result == 0) {
        return json{
            {"success", true},
            {"filename", filename},
            {"loaded_at", get_current_timestamp()}
        };
    } else {
        return json{
            {"success", false},
            {"error", "Failed to load memory"},
            {"error_code", result}
        };
    }
}
```

### 5. Real-time Statistics and Monitoring

#### Statistics Collection
```c
// Background thread for statistics collection
void* atlas_stats_collector_thread(void* arg) {
    struct llama_server_context* ctx = (struct llama_server_context*)arg;
    
    while (ctx->server_running) {
        if (ctx->atlas_enabled) {
            // Update statistics
            struct atlas_real_time_stats stats;
            llama_atlas_get_real_time_stats(&ctx->atlas_ctx, &stats);
            
            // Update server context
            ctx->atlas_stats.avg_memory_utilization = 
                (ctx->atlas_stats.avg_memory_utilization * 0.9) + 
                (stats.memory_utilization * 0.1);
            
            ctx->atlas_stats.avg_window_fill_ratio = 
                (ctx->atlas_stats.avg_window_fill_ratio * 0.9) + 
                (stats.window_fill_ratio * 0.1);
            
            // Log if enabled
            if (ctx->atlas_config.log_atlas_operations) {
                LOG_DEBUG("ATLAS Stats: Memory=%.2f%%, Window=%.2f%%, Loss=%.6f\n",
                         stats.memory_utilization * 100,
                         stats.window_fill_ratio * 100,
                         stats.current_omega_loss);
            }
        }
        
        usleep(ctx->atlas_config.stats_update_interval * 1000);
    }
    
    return NULL;
}

// GET /v1/atlas/stats
json handle_atlas_stats_get(struct llama_server_context* ctx) {
    if (!ctx->atlas_enabled) {
        return json{{"error", "ATLAS not enabled"}};
    }
    
    return json{
        {"requests_processed", ctx->atlas_stats.requests_processed},
        {"avg_memory_utilization", ctx->atlas_stats.avg_memory_utilization},
        {"avg_window_fill_ratio", ctx->atlas_stats.avg_window_fill_ratio},
        {"total_muon_iterations", ctx->atlas_stats.muon_iterations_total},
        {"total_omega_loss", ctx->atlas_stats.total_omega_loss},
        {"uptime_seconds", get_server_uptime()},
        {"memory_saves", get_memory_save_count()},
        {"memory_loads", get_memory_load_count()}
    };
}
```

## Testing Requirements

### API Testing
- [ ] **Completion endpoint**: ATLAS parameters in requests work correctly
- [ ] **Memory management**: Save/load memory via API endpoints
- [ ] **Statistics endpoints**: Real-time stats and monitoring data
- [ ] **Configuration**: Runtime ATLAS parameter changes
- [ ] **Error handling**: Graceful failures when ATLAS unavailable

### Integration Testing
- [ ] **Concurrent requests**: Multiple clients with different ATLAS configs
- [ ] **Memory persistence**: Server restart preserves ATLAS memory
- [ ] **Performance**: API response times with ATLAS enabled
- [ ] **Resource management**: Memory cleanup on server shutdown
- [ ] **Backward compatibility**: Non-ATLAS clients unaffected

### Load Testing
- [ ] **High throughput**: 100+ concurrent ATLAS requests
- [ ] **Memory scaling**: Large context requests (32K+ tokens)
- [ ] **Long-running**: 24+ hour server operation with ATLAS
- [ ] **Memory growth**: No memory leaks during extended operation

## Implementation Files

### Server Extensions
- `examples/server/server-atlas.cpp` - ATLAS server implementation
- `examples/server/server-atlas.h` - ATLAS server headers
- `examples/server/atlas-endpoints.cpp` - ATLAS-specific endpoints
- `examples/server/atlas-stats.cpp` - Statistics collection and monitoring

### API Documentation
- `examples/server/README-atlas.md` - ATLAS API documentation
- `examples/server/atlas-openapi.yaml` - OpenAPI specification
- `examples/server/atlas-examples.json` - Example API requests

### Configuration
- `examples/server/atlas-config.json` - Default ATLAS configuration
- `examples/server/atlas-docker/` - Docker configuration with ATLAS

### Test Files
- `tests/atlas/test-server-api.cpp` - API endpoint testing
- `tests/atlas/test-server-integration.cpp` - Integration testing
- `tests/atlas/test-server-performance.cpp` - Performance validation

## Success Criteria

### Functional Requirements
- [ ] All ATLAS features accessible via HTTP API
- [ ] Memory persistence works across server restarts
- [ ] Real-time statistics and monitoring functional
- [ ] Concurrent client support with independent ATLAS configs
- [ ] Graceful error handling and recovery

### Performance Requirements
- [ ] API latency overhead <50ms for ATLAS operations
- [ ] Support 50+ concurrent ATLAS clients
- [ ] Memory usage scales linearly with clients
- [ ] No performance degradation for non-ATLAS requests

### API Requirements
- [ ] RESTful design following OpenAPI 3.0 standards
- [ ] Comprehensive error responses with HTTP status codes
- [ ] Request/response validation and sanitization
- [ ] Rate limiting and authentication ready

## Dependencies
- Issues #3-9: All ATLAS components implemented
- llama-server architecture and HTTP framework
- JSON parsing library (nlohmann/json)
- HTTP server framework (existing llama-server)

## Configuration Example

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "threads": 8
  },
  "atlas": {
    "enabled": true,
    "auto_load_memory": true,
    "auto_save_memory": true,
    "memory_save_interval": 300,
    "enable_memory_endpoints": true,
    "enable_realtime_stats": true,
    "log_atlas_operations": false,
    "default_config": {
      "window_size": 512,
      "blend_ratio": 0.5,
      "learning_rate": 0.001,
      "polynomial_degree": 3,
      "use_muon": true,
      "warmup_tokens": 128
    }
  }
}
```

## Estimated Effort
**2-3 weeks** for experienced web API developer with llama.cpp knowledge

## References
- llama-server architecture documentation
- HTTP API design best practices
- OpenAPI 3.0 specification
- RESTful API guidelines
