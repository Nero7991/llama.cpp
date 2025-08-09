# ATLAS Integration Guide

This document provides a comprehensive guide for integrating ATLAS (Advanced Tensor Learning and Attention System) Phase 6A into llama.cpp server applications.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)  
- [Build Configuration](#build-configuration)
- [Integration Approaches](#integration-approaches)
- [Configuration Options](#configuration-options)
- [API Usage](#api-usage)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Observability](#monitoring-and-observability)
- [Error Handling](#error-handling)
- [Advanced Topics](#advanced-topics)
- [Migration Guide](#migration-guide)
- [Troubleshooting](#troubleshooting)

## Overview

ATLAS Phase 6A seamlessly integrates with llama-server to provide:

### Core Features
- **Thread-Safe Context Management**: Manages multiple llama contexts with sub-5ms overhead
- **Memory Persistence**: Automatic conversation state saving and loading  
- **Advanced Metrics**: Real-time performance monitoring with detailed analytics
- **OpenAI Compatibility**: 100% backward compatibility with OpenAI APIs
- **High Concurrency**: Supports 32+ simultaneous requests efficiently
- **Smart Caching**: Multiple caching strategies for optimal performance

### Performance Benefits  
- **Low Latency**: <5ms API overhead for ATLAS-enhanced requests
- **High Throughput**: Optimized concurrent request processing
- **Memory Efficient**: Intelligent memory pooling and persistence
- **Scalable**: Linear performance scaling with additional contexts

## Quick Start

### 1. Build with ATLAS
```bash
# Clone and configure
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure with ATLAS enabled
cmake -B build -DLLAMA_ATLAS=ON -DCMAKE_BUILD_TYPE=Release

# Build the server  
cmake --build build --target llama-server

# Verify ATLAS integration
./build/tools/server/llama-server --help | grep -i atlas
```

### 2. Start the Server
```bash
# Basic server with ATLAS
./build/tools/server/llama-server \
  --model your_model.gguf \
  --host 0.0.0.0 \
  --port 8080

# With ATLAS-specific options
export ATLAS_ENABLED=1
export ATLAS_MAX_CONCURRENT=32
export ATLAS_MEMORY_PATH="/tmp/atlas_memory"
./build/tools/server/llama-server --model your_model.gguf
```

### 3. Test Integration
```bash
# Standard OpenAI request (works unchanged)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# ATLAS-enhanced request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model", 
    "messages": [{"role": "user", "content": "Hello!"}],
    "atlas": {
      "memory_layers": 3,
      "session_id": "test_session",
      "cache_strategy": "adaptive"
    }
  }'

# Check ATLAS status
curl http://localhost:8080/v1/atlas/status
```

## Build Configuration

### CMake Options
```cmake
# Essential ATLAS options
option(LLAMA_ATLAS "Enable ATLAS integration" OFF)

# Additional related options
option(LLAMA_BUILD_TESTS "Build tests" OFF)         # For ATLAS tests
option(LLAMA_CUDA "Enable CUDA support" ON)        # Recommended with ATLAS
option(LLAMA_OPENMP "Enable OpenMP" ON)            # For parallel processing
```

### Build Examples
```bash
# Minimal ATLAS build
cmake -B build -DLLAMA_ATLAS=ON

# Production build with optimizations
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_ATLAS=ON \
  -DLLAMA_CUDA=ON \
  -DLLAMA_OPENMP=ON \
  -DLLAMA_NATIVE=ON

# Development build with tests
cmake -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLAMA_ATLAS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_CUDA=ON

# Cross-compilation example
cmake -B build \
  -DCMAKE_TOOLCHAIN_FILE=android.toolchain.cmake \
  -DLLAMA_ATLAS=ON \
  -DANDROID_ABI=arm64-v8a
```

### Verify Build
```bash
# Check if ATLAS was compiled in
strings build/tools/server/llama-server | grep -i atlas

# Run built-in tests
ctest --test-dir build -R atlas

# Check dependencies
ldd build/tools/server/llama-server | grep -E "(thread|ssl)"
```

## Integration Approaches  

### 1. Simple Integration (Recommended for Most Users)

Use the convenience wrapper for minimal code changes:

```cpp
#include "examples/server/atlas-integration.hpp"

int main() {
    // Initialize llama model
    llama_model* model = llama_load_model_from_file("model.gguf", params);
    
    // Initialize ATLAS with model
    atlas::init_atlas(model);
    
    // In your request handler
    auto handle_request = [](const json& request) -> json {
        // Check if request has ATLAS parameters
        if (atlas::is_atlas_request(request)) {
            // Process with ATLAS enhancements  
            auto response = atlas::process_with_atlas(request);
            return atlas::enhance_response_with_atlas(response, true);
        }
        
        // Fall back to standard processing
        return process_standard_request(request);
    };
    
    return 0;
}
```

### 2. HTTP Server Integration

Add ATLAS endpoints to your HTTP server:

```cpp
#include "examples/server/atlas-integration.hpp"
#include <httplib.h>

int main() {
    httplib::Server server;
    llama_model* model = /* initialize model */;
    
    // Initialize ATLAS
    atlas::init_atlas(model);
    
    // Add ATLAS endpoints using macro
    ATLAS_ADD_ENDPOINTS(server);
    
    // Enhanced chat completions
    server.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
        json request = json::parse(req.body);
        json response;
        
        // Process request (ATLAS or standard)
        ATLAS_PROCESS_REQUEST(request, response);
        
        res.set_content(response.dump(), "application/json");
    });
    
    server.listen("0.0.0.0", 8080);
    return 0;
}
```

### 3. Advanced Integration

Full control over ATLAS components:

```cpp
#include "examples/server/atlas-server.hpp"

class CustomAtlasServer {
private:
    std::unique_ptr<atlas::AtlasServer> atlas_server;
    atlas::AtlasServerConfig config;
    
public:
    bool initialize(llama_model* model) {
        // Configure ATLAS
        config.enable_memory_persistence = true;
        config.memory_save_path = "/var/lib/atlas/memory";
        config.max_concurrent_atlas_requests = 64;
        config.api_latency_budget_ms = 3.0f;  // Stricter latency
        config.enable_metrics_collection = true;
        
        // Initialize ATLAS server
        atlas_server = std::make_unique<atlas::AtlasServer>(config);
        return atlas_server->initialize(model);
    }
    
    json process_completion(const json& request) {
        // Validate request
        std::string error;
        if (!atlas::validation::validate_completion_request(request, error)) {
            return atlas::error_handling::create_validation_error("request", error);
        }
        
        // Process with ATLAS
        return atlas_server->handle_completion_request(request);
    }
    
    json get_metrics() {
        return atlas_server->handle_metrics_request();
    }
    
    void shutdown() {
        if (atlas_server) {
            atlas_server->shutdown();
        }
    }
};
```

## Configuration Options

### Environment Variables
```bash
# Core ATLAS settings
export ATLAS_ENABLED=1
export ATLAS_MAX_CONCURRENT=32
export ATLAS_LATENCY_BUDGET_MS=5.0

# Memory persistence
export ATLAS_MEMORY_PATH="/var/lib/atlas/memory"  
export ATLAS_AUTO_SAVE_INTERVAL=300  # seconds
export ATLAS_MEMORY_MAX_SIZE=1073741824  # 1GB

# Performance tuning
export ATLAS_CONTEXT_POOL_SIZE=16
export ATLAS_METRICS_INTERVAL=1000  # milliseconds
export ATLAS_CACHE_SIZE_L1=67108864  # 64MB
export ATLAS_CACHE_SIZE_L2=268435456  # 256MB

# Logging and debugging
export ATLAS_LOG_LEVEL=INFO  # DEBUG, INFO, WARN, ERROR
export ATLAS_VERBOSE_METRICS=0
export ATLAS_ENABLE_PROFILING=0
```

### Runtime Configuration
```cpp
atlas::AtlasServerConfig config;

// Core settings
config.enable_memory_persistence = true;
config.enable_metrics_collection = true;
config.enable_batch_processing = true;

// Performance
config.max_concurrent_atlas_requests = 32;
config.api_latency_budget_ms = 5.0f;
config.context_pool_size = 16;

// Memory management
config.memory_save_path = "/var/lib/atlas/memory";
config.memory_auto_save_interval = 300;  // seconds
config.l1_cache_size = 64 * 1024 * 1024;   // 64MB
config.l2_cache_size = 256 * 1024 * 1024;  // 256MB
config.l3_cache_size = 1024 * 1024 * 1024; // 1GB

// Metrics
config.metrics_update_interval_ms = 1000;
config.metrics_history_size = 1000;
```

### Dynamic Configuration
```cpp
// Update configuration at runtime
atlas::AtlasServerConfig new_config = current_config;
new_config.max_concurrent_atlas_requests = 64;
atlas_server->update_config(new_config);

// Get current configuration
auto current = atlas_server->get_config();
std::cout << "Current max concurrent: " << current.max_concurrent_atlas_requests << std::endl;
```

## API Usage

### Basic Chat Completion with ATLAS
```cpp
// C++ example
json request = {
    {"model", "llama-2-7b"},
    {"messages", {
        {{"role", "user"}, {"content", "What is machine learning?"}}
    }},
    {"atlas", {
        {"memory_layers", 3},
        {"session_id", "user123_ml_session"},
        {"cache_strategy", "adaptive"}
    }}
};

auto response = atlas_server->handle_completion_request(request);
```

```python
# Python example
import requests

response = requests.post('http://localhost:8080/v1/chat/completions', json={
    'model': 'llama-2-7b',
    'messages': [
        {'role': 'user', 'content': 'What is machine learning?'}
    ],
    'atlas': {
        'memory_layers': 3,
        'session_id': 'user123_ml_session', 
        'cache_strategy': 'adaptive'
    }
})
```

### Memory Persistence
```cpp
// Save conversation context
json save_request = {
    {"context_id", "user123_conversation"},
    {"state", {
        {"conversation_history", {/* previous messages */}},
        {"user_preferences", {{"response_style", "detailed"}}},
        {"session_metadata", {{"topic", "machine_learning"}}}
    }}
};

auto save_response = atlas_server->handle_memory_operation("save", save_request);

// Load context in subsequent session
json load_request = {{"context_id", "user123_conversation"}};
auto load_response = atlas_server->handle_memory_operation("load", load_request);
```

### Real-Time Metrics
```cpp
// Get current metrics
auto metrics = atlas_server->handle_metrics_request();

// Extract key performance indicators
double avg_latency = metrics["data"]["current"]["latency"]["avg_ms"];
int active_requests = metrics["data"]["current"]["active_requests"];
double cache_hit_ratio = metrics["data"]["current"]["cache"]["hit_ratio"];

std::cout << "Performance: " << avg_latency << "ms avg latency, "
          << active_requests << " active, "
          << (cache_hit_ratio * 100) << "% cache hits" << std::endl;
```

## Performance Optimization

### Context Pool Sizing
```cpp
// Calculate optimal pool size
int hardware_threads = std::thread::hardware_concurrency();
int expected_concurrent_users = 100;
int avg_request_duration_ms = 500;
int target_response_time_ms = 1000;

// Pool size calculation
int optimal_pool_size = std::min(
    hardware_threads * 2,  // Don't exceed 2x CPU cores
    (expected_concurrent_users * avg_request_duration_ms) / target_response_time_ms
);

config.context_pool_size = optimal_pool_size;
config.max_concurrent_atlas_requests = optimal_pool_size * 2;
```

### Memory Optimization
```cpp
// Optimize memory usage based on available RAM
size_t available_memory = get_available_system_memory();
size_t model_memory = get_model_memory_usage();
size_t available_for_atlas = available_memory - model_memory;

// Distribute cache memory hierarchically
config.l1_cache_size = std::min(available_for_atlas / 16, 64ULL * 1024 * 1024);
config.l2_cache_size = std::min(available_for_atlas / 8, 256ULL * 1024 * 1024);  
config.l3_cache_size = std::min(available_for_atlas / 4, 1024ULL * 1024 * 1024);
```

### Latency Optimization
```cpp
// Configure for minimal latency
config.api_latency_budget_ms = 2.0f;  // Aggressive latency target
config.enable_batch_processing = false;  // Reduce batching overhead
config.metrics_update_interval_ms = 5000;  // Less frequent metrics

// Use appropriate cache strategy
// - "lru" for temporal locality
// - "lfu" for frequency patterns
// - "adaptive" for mixed workloads
```

### Throughput Optimization
```cpp
// Configure for maximum throughput
config.max_concurrent_atlas_requests = hardware_threads * 4;
config.enable_batch_processing = true;
config.api_latency_budget_ms = 10.0f;  // More relaxed latency

// Optimize for bulk processing
config.context_pool_size = hardware_threads * 2;
config.metrics_update_interval_ms = 10000;  // Less frequent updates
```

## Monitoring and Observability

### Key Metrics to Monitor
```cpp
// Extract and monitor critical metrics
auto metrics = atlas_server->handle_metrics_request();
auto current = metrics["data"]["current"];

// Latency metrics (target: <5ms average)
double avg_latency = current["latency"]["avg_ms"];
double p95_latency = current["latency"]["p95_ms"];
double p99_latency = current["latency"]["p99_ms"];

// Throughput metrics  
double requests_per_second = current["throughput"]["requests_per_second"];
double tokens_per_second = current["throughput"]["tokens_per_second"];

// Resource utilization
int active_requests = current["active_requests"];
int max_concurrent = config.max_concurrent_atlas_requests;
double utilization = static_cast<double>(active_requests) / max_concurrent;

// Cache performance
double cache_hit_ratio = current["cache"]["hit_ratio"];
int cache_hits = current["cache"]["hits"];
int cache_misses = current["cache"]["misses"];

// Memory usage
long memory_bytes = current["memory"]["current_bytes"];
long peak_memory = current["memory"]["peak_bytes"];
double memory_mb = memory_bytes / (1024.0 * 1024.0);
```

### Alerting Configuration
```cpp
// Define alert thresholds
struct AtlasAlerts {
    double max_avg_latency_ms = 5.0;
    double max_p95_latency_ms = 10.0;
    double max_utilization_percent = 80.0;
    double min_cache_hit_ratio = 0.7;
    double max_error_rate_percent = 5.0;
    long max_memory_mb = 4096;
};

// Check alerts
bool check_alerts(const json& metrics, const AtlasAlerts& alerts) {
    auto current = metrics["data"]["current"];
    
    if (current["latency"]["avg_ms"] > alerts.max_avg_latency_ms) {
        log_alert("High average latency: " + std::to_string(current["latency"]["avg_ms"]) + "ms");
        return false;
    }
    
    double utilization = current["active_requests"] / static_cast<double>(config.max_concurrent_atlas_requests);
    if (utilization > alerts.max_utilization_percent / 100.0) {
        log_alert("High utilization: " + std::to_string(utilization * 100) + "%");
        return false;
    }
    
    // Add more alert checks...
    return true;
}
```

### Prometheus Integration
```cpp
// Export metrics in Prometheus format
std::string export_prometheus_metrics(const json& metrics) {
    std::ostringstream oss;
    auto current = metrics["data"]["current"];
    
    // Latency metrics
    oss << "# HELP atlas_latency_seconds Request latency in seconds\n";
    oss << "# TYPE atlas_latency_seconds histogram\n";
    oss << "atlas_latency_seconds_sum " << (current["latency"]["avg_ms"] / 1000.0) << "\n";
    oss << "atlas_latency_seconds_count " << current["completed_requests"] << "\n";
    
    // Throughput metrics
    oss << "# HELP atlas_requests_per_second Current requests per second\n";
    oss << "# TYPE atlas_requests_per_second gauge\n";
    oss << "atlas_requests_per_second " << current["throughput"]["requests_per_second"] << "\n";
    
    // Resource utilization
    oss << "# HELP atlas_active_requests Currently active requests\n";
    oss << "# TYPE atlas_active_requests gauge\n";
    oss << "atlas_active_requests " << current["active_requests"] << "\n";
    
    return oss.str();
}
```

### Real-Time Dashboards
```javascript
// JavaScript dashboard integration
class AtlasDashboard {
    constructor(atlasUrl) {
        this.atlasUrl = atlasUrl;
        this.metricsSource = new EventSource(`${atlasUrl}/v1/atlas/metrics/stream`);
        
        this.metricsSource.onmessage = (event) => {
            const metrics = JSON.parse(event.data);
            this.updateDashboard(metrics);
        };
    }
    
    updateDashboard(metrics) {
        // Update latency chart
        this.latencyChart.addPoint(metrics.timestamp, metrics.avg_latency);
        
        // Update throughput gauge
        this.throughputGauge.setValue(metrics.rps);
        
        // Update active requests counter
        document.getElementById('active-requests').textContent = metrics.active_requests;
        
        // Update cache hit ratio
        const hitRatio = (metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)) * 100;
        document.getElementById('cache-hit-ratio').textContent = `${hitRatio.toFixed(1)}%`;
    }
}

// Initialize dashboard
const dashboard = new AtlasDashboard('http://localhost:8080');
```

## Error Handling

### Comprehensive Error Handling
```cpp
class AtlasErrorHandler {
public:
    static json handle_request_safely(
        std::function<json()> request_handler,
        const std::string& operation_name
    ) {
        try {
            return request_handler();
        }
        catch (const atlas::validation_error& e) {
            return atlas::error_handling::create_validation_error(
                e.field(), e.what()
            );
        }
        catch (const atlas::resource_exhausted& e) {
            return atlas::error_handling::create_error_response(
                503, "Service temporarily unavailable: " + std::string(e.what()),
                "resource_exhausted"
            );
        }
        catch (const atlas::timeout_error& e) {
            return atlas::error_handling::create_error_response(
                408, "Request timeout: " + std::string(e.what()),
                "timeout"
            );
        }
        catch (const std::exception& e) {
            // Log the full error for debugging
            log_error("ATLAS operation '" + operation_name + "' failed: " + e.what());
            
            return atlas::error_handling::create_internal_error(
                "Operation failed"  // Don't expose internal details
            );
        }
    }
};

// Usage example
json process_completion_safely(const json& request) {
    return AtlasErrorHandler::handle_request_safely([&]() {
        return atlas_server->handle_completion_request(request);
    }, "completion_request");
}
```

### Graceful Degradation
```cpp
// Fallback to standard processing when ATLAS fails
json handle_chat_completion(const json& request) {
    // Try ATLAS first if enabled and request has ATLAS parameters
    if (atlas_enabled && atlas::is_atlas_request(request)) {
        try {
            auto atlas_response = atlas_server->handle_completion_request(request);
            
            // Check if ATLAS processing was successful
            if (atlas_response.contains("atlas") && 
                atlas_response["atlas"]["processed"] == true) {
                return atlas_response;
            }
        }
        catch (const std::exception& e) {
            // Log ATLAS failure but continue with standard processing
            log_warning("ATLAS processing failed, falling back to standard: " + std::string(e.what()));
        }
    }
    
    // Fall back to standard llama processing
    return handle_standard_completion(request);
}
```

### Circuit Breaker Pattern
```cpp
class AtlasCircuitBreaker {
private:
    std::atomic<int> failure_count{0};
    std::atomic<bool> circuit_open{false};
    std::chrono::steady_clock::time_point last_failure_time;
    
    static constexpr int FAILURE_THRESHOLD = 5;
    static constexpr auto RECOVERY_TIMEOUT = std::chrono::minutes(1);
    
public:
    bool should_allow_request() {
        if (!circuit_open.load()) {
            return true;  // Circuit closed, allow requests
        }
        
        // Check if recovery timeout has passed
        auto now = std::chrono::steady_clock::now();
        if (now - last_failure_time > RECOVERY_TIMEOUT) {
            circuit_open = false;
            failure_count = 0;
            return true;
        }
        
        return false;  // Circuit open, reject requests
    }
    
    void record_success() {
        failure_count = 0;
        circuit_open = false;
    }
    
    void record_failure() {
        failure_count++;
        last_failure_time = std::chrono::steady_clock::now();
        
        if (failure_count >= FAILURE_THRESHOLD) {
            circuit_open = true;
        }
    }
};
```

## Advanced Topics

### Custom Metrics Collection
```cpp
class CustomMetricsCollector : public atlas::MetricsCollector {
public:
    void record_custom_metric(const std::string& name, double value) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        custom_metrics_[name].add_value(value);
    }
    
    json get_custom_metrics() const override {
        json custom = json::object();
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        for (const auto& [name, window] : custom_metrics_) {
            custom[name] = {
                {"avg", window.average()},
                {"min", window.min()},
                {"max", window.max()},
                {"count", window.size()}
            };
        }
        
        return custom;
    }
    
private:
    mutable std::mutex metrics_mutex_;
    std::unordered_map<std::string, atlas::metrics::sliding_window<double>> custom_metrics_;
};
```

### Plugin Architecture
```cpp
class AtlasPlugin {
public:
    virtual ~AtlasPlugin() = default;
    virtual bool initialize(const json& config) = 0;
    virtual json process_request(const json& request) = 0;
    virtual void shutdown() = 0;
};

class AtlasPluginManager {
private:
    std::vector<std::unique_ptr<AtlasPlugin>> plugins_;
    
public:
    void register_plugin(std::unique_ptr<AtlasPlugin> plugin) {
        plugins_.push_back(std::move(plugin));
    }
    
    json process_with_plugins(const json& request) {
        json result = request;
        
        for (auto& plugin : plugins_) {
            result = plugin->process_request(result);
        }
        
        return result;
    }
};
```

### Multi-Model Support
```cpp
class MultiModelAtlasServer {
private:
    std::unordered_map<std::string, std::unique_ptr<atlas::AtlasServer>> model_servers_;
    
public:
    bool add_model(const std::string& model_name, llama_model* model) {
        atlas::AtlasServerConfig config;
        config.max_concurrent_atlas_requests = 16;  // Per model
        
        auto server = std::make_unique<atlas::AtlasServer>(config);
        if (!server->initialize(model)) {
            return false;
        }
        
        model_servers_[model_name] = std::move(server);
        return true;
    }
    
    json handle_request(const json& request) {
        std::string model_name = request.value("model", "default");
        
        auto it = model_servers_.find(model_name);
        if (it == model_servers_.end()) {
            return atlas::error_handling::create_error_response(
                404, "Model not found: " + model_name, "model_not_found"
            );
        }
        
        return it->second->handle_completion_request(request);
    }
};
```

## Migration Guide

### From Standard llama-server

#### 1. Update Build Configuration
```bash
# Before: Standard build
cmake -B build

# After: ATLAS-enabled build
cmake -B build -DLLAMA_ATLAS=ON
```

#### 2. Update Server Code
```cpp
// Before: Basic server
int main() {
    llama_model* model = load_model();
    
    httplib::Server server;
    server.Post("/v1/chat/completions", [model](auto& req, auto& res) {
        auto request = json::parse(req.body);
        auto response = process_completion(model, request);
        res.set_content(response.dump(), "application/json");
    });
    
    server.listen("0.0.0.0", 8080);
}

// After: ATLAS-enhanced server  
int main() {
    llama_model* model = load_model();
    
    // Initialize ATLAS
    atlas::init_atlas(model);
    
    httplib::Server server;
    server.Post("/v1/chat/completions", [model](auto& req, auto& res) {
        auto request = json::parse(req.body);
        json response;
        
        // Process with ATLAS if applicable, otherwise standard processing
        ATLAS_PROCESS_REQUEST(request, response);
        
        res.set_content(response.dump(), "application/json");
    });
    
    // Add ATLAS-specific endpoints
    ATLAS_ADD_ENDPOINTS(server);
    
    server.listen("0.0.0.0", 8080);
}
```

#### 3. Update Client Code (Optional)
```python
# Before: Standard client
response = requests.post('/v1/chat/completions', json={
    'model': 'llama-2-7b',
    'messages': [{'role': 'user', 'content': 'Hello'}]
})

# After: ATLAS-enhanced client (optional)
response = requests.post('/v1/chat/completions', json={
    'model': 'llama-2-7b',
    'messages': [{'role': 'user', 'content': 'Hello'}],
    'atlas': {                          # Optional ATLAS parameters
        'session_id': 'user123',        # For conversation continuity
        'memory_layers': 3,             # Enhanced context handling
        'cache_strategy': 'adaptive'    # Optimized caching
    }
})
```

#### 4. Update Configuration
```bash
# Add ATLAS environment variables
export ATLAS_ENABLED=1
export ATLAS_MAX_CONCURRENT=32
export ATLAS_MEMORY_PATH="/var/lib/atlas/memory"

# Update startup scripts
./llama-server --model model.gguf  # Works unchanged with ATLAS
```

### Backward Compatibility
- **100% API Compatibility**: All existing OpenAI-compatible requests work unchanged
- **Optional Enhancement**: ATLAS features are opt-in via request parameters
- **Graceful Fallback**: Falls back to standard processing if ATLAS fails
- **No Breaking Changes**: Existing clients continue working without modification

## Troubleshooting

### Common Build Issues

#### ATLAS Not Found During Build
```bash
# Error: ATLAS headers not found
# Solution: Ensure ATLAS source files exist
ls -la examples/server/atlas-*.hpp examples/server/atlas-*.cpp

# If missing, check git status
git status
git ls-files examples/server/atlas-*
```

#### Compilation Errors
```bash
# Error: C++17 features not available
# Solution: Ensure C++17 support
cmake -B build -DCMAKE_CXX_STANDARD=17 -DLLAMA_ATLAS=ON

# Error: Threading library not found
# Solution: Install threading support
sudo apt-get install libpthread-stubs0-dev  # Ubuntu/Debian
```

### Runtime Issues

#### High Latency
```cpp
// Check context pool utilization
auto metrics = atlas_server->get_metrics();
int active = metrics["data"]["current"]["active_requests"];
int max_concurrent = config.max_concurrent_atlas_requests;
double utilization = static_cast<double>(active) / max_concurrent;

if (utilization > 0.8) {
    // Increase context pool size
    config.max_concurrent_atlas_requests *= 2;
    atlas_server->update_config(config);
}
```

#### Memory Issues
```bash
# Monitor memory usage
valgrind --tool=massif ./llama-server --model model.gguf

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./llama-server

# Monitor runtime memory
watch -n 1 'ps -p $(pgrep llama-server) -o pid,vsz,rss,pmem'
```

#### Context Pool Exhaustion  
```cpp
// Add monitoring for pool exhaustion
auto handle_request = [](const json& request) -> json {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Try to acquire context with timeout
    auto ctx = atlas_server->get_context_pool().acquire_context(
        std::chrono::milliseconds(1000)  // 1 second timeout
    );
    
    if (!ctx) {
        return atlas::error_handling::create_error_response(
            503, "No available ATLAS contexts. Please try again later.",
            "resource_exhausted"
        );
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto wait_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    );
    
    if (wait_time.count() > 100) {  // Log if waited >100ms
        log_warning("Context acquisition took " + std::to_string(wait_time.count()) + "ms");
    }
    
    // Process request...
};
```

### Performance Debugging

#### Profiling with perf
```bash
# Record performance data
perf record -g --call-graph=dwarf ./llama-server --model model.gguf

# Analyze results
perf report --stdio | head -50

# Focus on ATLAS functions
perf report --stdio | grep -E "(atlas|ATLAS)"
```

#### Memory Profiling
```bash
# Track memory allocations
heaptrack ./llama-server --model model.gguf

# Analyze heap usage
heaptrack_print heaptrack.llama-server.*.gz | head -50
```

#### Thread Analysis
```bash
# Monitor thread usage
top -H -p $(pgrep llama-server)

# Check thread safety with sanitizers
cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=thread" -DLLAMA_ATLAS=ON
cmake --build build
./build/tools/server/llama-server --model model.gguf
```

### Support Resources

#### Debug Information
When reporting issues, include:

```bash
# System information
uname -a
cat /proc/cpuinfo | grep "model name" | head -1
free -h
df -h

# Build information
cmake --version
gcc --version || clang --version
strings build/tools/server/llama-server | grep -i atlas

# Runtime information
./build/tools/server/llama-server --version
curl http://localhost:8080/v1/atlas/status

# Configuration
env | grep ATLAS
```

#### Log Collection
```bash
# Enable verbose logging
export ATLAS_LOG_LEVEL=DEBUG
export ATLAS_VERBOSE_METRICS=1

# Collect logs
./llama-server --model model.gguf 2>&1 | tee atlas_debug.log

# Filter ATLAS-specific logs
grep -E "(ATLAS|atlas)" atlas_debug.log
```

For additional support, consult:
- [ATLAS Test Suite Documentation](../tests/README_ATLAS_TESTS.md)
- [ATLAS API Reference](../examples/server/ATLAS_API.md)
- [llama.cpp Issues](https://github.com/ggml-org/llama.cpp/issues)

---

This integration guide provides comprehensive coverage for implementing ATLAS Phase 6A in production environments. For the most up-to-date information, refer to the official llama.cpp documentation and ATLAS-specific files in the repository.