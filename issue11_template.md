## Summary

Implement comprehensive llama-server API support for ATLAS-enhanced models, enabling REST API access to ATLAS test-time memorization capabilities with full configuration control and memory persistence.

## Background

With ATLAS core components implemented, llama-server needs API endpoints to:
- Configure ATLAS parameters via HTTP requests
- Monitor ATLAS memory usage and performance in real-time
- Control test-time memorization behavior through API parameters
- Provide seamless integration with existing OpenAI-compatible endpoints

## Implementation Requirements

### 1. ATLAS-Enhanced API Endpoints

#### Extended Completions Endpoint
```json
POST /v1/completions
{
  "prompt": "Long context document...",
  "max_tokens": 512,
  "atlas": {
    "enabled": true,
    "window_size": 1024,
    "blend_ratio": 0.7,
    "learning_rate": 0.001,
    "polynomial_degree": 3,
    "use_muon": true,
    "memory_file": "custom_memory.atlas",
    "save_memory": true
  }
}
```

#### ATLAS Configuration Endpoint
```json
GET /v1/atlas/config
{
  "enabled": true,
  "default_window_size": 512,
  "max_window_size": 4096,
  "supported_features": ["polynomial", "exponential", "muon"],
  "memory_capacity_mb": 2048,
  "current_memory_usage_mb": 1024
}

POST /v1/atlas/config
{
  "default_blend_ratio": 0.5,
  "auto_save_memory": true,
  "memory_save_interval": 300,
  "adaptive_window": true
}
```

#### ATLAS Status and Monitoring
```json
GET /v1/atlas/status
{
  "memory_modules": {
    "total_parameters": 2097152,
    "memory_usage_mb": 1024,
    "utilization_percent": 87.3
  },
  "sliding_window": {
    "current_size": 1024,
    "fill_ratio": 0.92,
    "oldest_token_age": 2048
  },
  "optimization": {
    "total_updates": 15429,
    "avg_loss": 0.023,
    "convergence_rate": 0.94,
    "muon_iterations": 4
  },
  "performance": {
    "tokens_per_second": 23.7,
    "memory_bandwidth_gbps": 12.4,
    "atlas_overhead_percent": 15.2
  }
}
```

#### Memory Management Endpoints
```json
POST /v1/atlas/memory/save
{
  "filename": "session_memory.atlas",
  "include_metadata": true
}

POST /v1/atlas/memory/load
{
  "filename": "session_memory.atlas",
  "reset_optimizer_state": false
}

GET /v1/atlas/memory/info
{
  "current_file": "llama-7b-chat_memory.atlas",
  "last_saved": "2025-08-08T20:30:45Z",
  "memory_size_mb": 245.6,
  "auto_save_enabled": true
}

DELETE /v1/atlas/memory/reset
{
  "reset_type": "soft"  // "soft" or "hard"
}
```

### 2. Server Implementation Integration

#### Enhanced Server Initialization
```cpp
// Enhanced server params with ATLAS support
struct llama_server_context {
    // ... existing fields ...
    
    // ATLAS-specific fields
    bool atlas_enabled;
    struct atlas_context atlas_ctx;
    std::string atlas_memory_file;
    bool atlas_auto_save;
    int atlas_save_interval;
    std::chrono::time_point<std::chrono::steady_clock> last_save_time;
    
    // ATLAS API handlers
    void handle_atlas_config(const httplib::Request& req, httplib::Response& res);
    void handle_atlas_status(const httplib::Request& req, httplib::Response& res);
    void handle_atlas_memory_save(const httplib::Request& req, httplib::Response& res);
    void handle_atlas_memory_load(const httplib::Request& req, httplib:Response& res);
};

// ATLAS-enhanced completion handler
void handle_completions_atlas(const httplib::Request& req, httplib::Response& res) {
    json request_data = json::parse(req.body);
    
    // Parse ATLAS parameters
    if (request_data.contains("atlas")) {
        auto atlas_config = request_data["atlas"];
        
        // Apply per-request ATLAS settings
        if (atlas_config.contains("window_size")) {
            ctx.atlas_ctx.window_size = atlas_config["window_size"];
        }
        if (atlas_config.contains("blend_ratio")) {
            ctx.atlas_ctx.blend_ratio = atlas_config["blend_ratio"];
        }
        // ... other parameters
        
        // Load custom memory file if specified
        if (atlas_config.contains("memory_file")) {
            std::string memory_file = atlas_config["memory_file"];
            atlas_load_memory(&ctx.atlas_ctx, memory_file.c_str());
        }
    }
    
    // Proceed with ATLAS-enhanced generation
    generate_with_atlas(request_data, res);
}
```

#### Real-time ATLAS Monitoring
```cpp
// ATLAS metrics collection
struct atlas_metrics {
    float tokens_per_second;
    float memory_usage_mb;
    float optimization_loss;
    int muon_iterations;
    float bandwidth_utilization;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

class AtlasMetricsCollector {
public:
    void collect_metrics(const atlas_context* ctx);
    atlas_metrics get_current_metrics();
    std::vector<atlas_metrics> get_history(int seconds);
    void reset_metrics();
    
private:
    std::deque<atlas_metrics> metrics_history;
    mutable std::mutex metrics_mutex;
};

// Background metrics collection thread
void atlas_metrics_thread(llama_server_context* ctx) {
    while (ctx->running) {
        ctx->metrics_collector.collect_metrics(&ctx->atlas_ctx);
        
        // Auto-save memory if interval exceeded
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - ctx->last_save_time).count();
            
        if (ctx->atlas_auto_save && elapsed >= ctx->atlas_save_interval) {
            atlas_save_memory_auto(&ctx->atlas_ctx, ctx->atlas_memory_file);
            ctx->last_save_time = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}
```

### 3. API Parameter Validation and Error Handling

#### Request Validation
```cpp
// ATLAS parameter validation
bool validate_atlas_params(const json& atlas_config, std::string& error) {
    if (atlas_config.contains("window_size")) {
        int window_size = atlas_config["window_size"];
        if (window_size < 64 || window_size > 8192) {
            error = "window_size must be between 64 and 8192";
            return false;
        }
    }
    
    if (atlas_config.contains("blend_ratio")) {
        float blend_ratio = atlas_config["blend_ratio"];
        if (blend_ratio < 0.0f || blend_ratio > 1.0f) {
            error = "blend_ratio must be between 0.0 and 1.0";
            return false;
        }
    }
    
    if (atlas_config.contains("learning_rate")) {
        float lr = atlas_config["learning_rate"];
        if (lr <= 0.0f || lr > 0.1f) {
            error = "learning_rate must be between 0.0 and 0.1";
            return false;
        }
    }
    
    return true;
}

// Error response formatting
void send_atlas_error(httplib::Response& res, const std::string& error, int code = 400) {
    json error_response = {
        {"error", {
            {"message", error},
            {"type", "atlas_parameter_error"},
            {"code", code}
        }}
    };
    res.status = code;
    res.set_content(error_response.dump(), "application/json");
}
```

### 4. OpenAI API Compatibility Extensions

#### Chat Completions with ATLAS
```json
POST /v1/chat/completions
{
  "model": "llama-7b-atlas",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze this 50-page document..."}
  ],
  "atlas": {
    "enabled": true,
    "window_size": 2048,
    "adaptive_blend": true
  }
}
```

#### Model Information with ATLAS Support
```json
GET /v1/models
{
  "data": [
    {
      "id": "llama-7b-atlas",
      "object": "model",
      "atlas_support": {
        "enabled": true,
        "max_context": 32768,
        "memory_capacity": 2048,
        "features": ["polynomial", "muon", "adaptive_window"]
      }
    }
  ]
}
```

## Testing Requirements

### API Integration Tests
- [ ] **Endpoint functionality**: All ATLAS API endpoints work correctly
- [ ] **Parameter validation**: Invalid parameters return appropriate errors
- [ ] **OpenAI compatibility**: ATLAS parameters don't break OpenAI API compliance
- [ ] **Concurrent requests**: Multiple clients can use ATLAS simultaneously
- [ ] **Memory persistence**: Save/load operations work correctly via API

### Performance Tests
- [ ] **API overhead**: ATLAS API calls add <5ms latency
- [ ] **Throughput**: Server maintains throughput with ATLAS enabled
- [ ] **Memory management**: No memory leaks during extended API usage
- [ ] **Auto-save performance**: Background saving doesn't impact response times

### Error Handling Tests
- [ ] **Invalid parameters**: Graceful error responses for all invalid inputs
- [ ] **Memory file errors**: Proper handling of corrupted/missing memory files
- [ ] **Resource exhaustion**: Proper responses when memory limits exceeded
- [ ] **Concurrent access**: Thread-safe ATLAS parameter updates

## Implementation Files

### Server Core
- `examples/server/atlas-server.cpp` - ATLAS-enhanced server implementation
- `examples/server/atlas-api.cpp` - ATLAS-specific API endpoints
- `examples/server/atlas-metrics.cpp` - Real-time metrics collection

### API Handlers
- `examples/server/handlers/atlas-config.cpp` - Configuration endpoint handlers
- `examples/server/handlers/atlas-memory.cpp` - Memory management endpoints
- `examples/server/handlers/atlas-status.cpp` - Status and monitoring endpoints

### Documentation
- `examples/server/README-ATLAS.md` - ATLAS server usage documentation
- `docs/atlas-api.md` - Complete API reference
- `examples/atlas-client/` - Example client implementations

### Test Files
- `tests/atlas/test-server-api.cpp` - API endpoint tests
- `tests/atlas/test-server-performance.cpp` - Performance validation
- `tests/atlas/test-server-concurrent.cpp` - Concurrency testing

## Success Criteria

### Functional Requirements
- [ ] All ATLAS API endpoints implemented and functional
- [ ] Full OpenAI API compatibility maintained with ATLAS extensions
- [ ] Memory persistence works reliably across server restarts
- [ ] Real-time monitoring provides accurate ATLAS metrics
- [ ] Parameter validation prevents invalid configurations

### Performance Requirements
- [ ] API latency overhead <5ms for ATLAS-enhanced requests
- [ ] Server maintains >95% of baseline throughput with ATLAS
- [ ] Memory auto-save operations complete in <100ms
- [ ] Concurrent request handling without performance degradation

### Quality Requirements
- [ ] 100% API test coverage for all ATLAS endpoints
- [ ] Zero memory leaks during 24+ hour stress testing
- [ ] Graceful error handling for all edge cases
- [ ] Complete API documentation with examples

## Dependencies
- Issues #3-10: All core ATLAS components implemented
- llama-server infrastructure
- httplib HTTP server library
- JSON parsing library (nlohmann/json)

## Estimated Effort
**2-3 weeks** for experienced API/server developer

## API Usage Examples

### Basic ATLAS-Enhanced Generation
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Summarize this long document...",
    "max_tokens": 512,
    "atlas": {
      "enabled": true,
      "window_size": 1024
    }
  }'
```

### Advanced Configuration
```bash
# Configure ATLAS defaults
curl -X POST http://localhost:8080/v1/atlas/config \
  -H "Content-Type: application/json" \
  -d '{
    "default_blend_ratio": 0.7,
    "auto_save_memory": true,
    "memory_save_interval": 600
  }'

# Monitor ATLAS performance
curl http://localhost:8080/v1/atlas/status

# Save current memory state
curl -X POST http://localhost:8080/v1/atlas/memory/save \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_session.atlas"}'
```

## References
- llama-server architecture documentation
- OpenAI API specification
- REST API best practices
- HTTP server performance optimization
