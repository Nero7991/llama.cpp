# ATLAS Integration for llama-server

## Overview

ATLAS (Advanced Tensor Learning and Attention System) Phase 6A provides seamless integration with llama-server, offering enhanced performance, memory persistence, and advanced metrics collection while maintaining full OpenAI API compatibility.

## Features

### Core Capabilities
- **Thread-Safe Context Pool**: Manages multiple llama contexts with <5ms overhead
- **Memory Persistence**: Automatic state saving and loading with JSON serialization
- **Advanced Metrics**: Real-time performance monitoring with sliding window analytics
- **OpenAI Compatibility**: Full compatibility with OpenAI Chat and Completion APIs
- **Real-Time Streaming**: Server-Sent Events for live metrics updates
- **Concurrent Processing**: Supports 32+ simultaneous requests

### Performance Benefits
- **Low Latency**: <5ms API overhead for ATLAS-enhanced requests
- **High Throughput**: Optimized for concurrent request handling
- **Memory Efficient**: Smart memory pool management and persistence
- **Lock-Free Metrics**: High-performance metrics collection with minimal overhead

## Quick Start

### Build Configuration
```bash
# Configure llama.cpp with ATLAS enabled
cmake -B build -DLLAMA_ATLAS=ON

# Build the server
cmake --build build --target llama-server

# Run with ATLAS integration
./build/tools/server/llama-server --model your_model.gguf
```

### Basic Usage
```bash
# Standard OpenAI-compatible request
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "atlas": {
      "memory_layers": 3,
      "cache_strategy": "adaptive",
      "session_id": "user123"
    }
  }'

# Check ATLAS status
curl http://localhost:8080/v1/atlas/status

# Get real-time metrics
curl http://localhost:8080/v1/atlas/metrics
```

## API Reference

### Enhanced OpenAI Endpoints

#### Chat Completions with ATLAS
**POST** `/v1/chat/completions`

Standard OpenAI request with optional ATLAS extensions:

```json
{
  "model": "your-model",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "atlas": {
    "memory_layers": 3,
    "cache_strategy": "adaptive",
    "session_id": "unique-session-id",
    "batch_size": 8
  }
}
```

**ATLAS Parameters:**
- `memory_layers` (1-10): Number of memory layers to utilize
- `cache_strategy`: "lru", "lfu", or "adaptive" caching strategy
- `session_id`: Persistent session identifier for memory continuity
- `batch_size` (1-32): Batch processing size for efficiency

#### Standard Completions
**POST** `/v1/completions`

Compatible with OpenAI completions API with ATLAS enhancements:

```json
{
  "model": "your-model",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "atlas": {
    "memory_layers": 2,
    "session_id": "completion-session"
  }
}
```

### ATLAS-Specific Endpoints

#### Health Check
**GET** `/v1/atlas/status`

Returns ATLAS system status and health information:

```json
{
  "status": "healthy",
  "data": {
    "initialized": true,
    "context_pool_size": 8,
    "metrics_enabled": true,
    "memory_persistence": true,
    "uptime_seconds": 3600
  }
}
```

#### Metrics Endpoint
**GET** `/v1/atlas/metrics`

Provides comprehensive performance metrics:

```json
{
  "status": "success",
  "data": {
    "current": {
      "total_requests": 1250,
      "active_requests": 3,
      "completed_requests": 1247,
      "failed_requests": 3,
      "avg_latency_ms": 2.3,
      "max_latency_ms": 15.2,
      "min_latency_ms": 0.8,
      "memory_usage_bytes": 524288000,
      "cache_hits": 890,
      "cache_misses": 357,
      "throughput_tokens_per_sec": 1250.5,
      "uptime_seconds": 3600
    },
    "history_size": 100,
    "server_health": true
  }
}
```

#### Real-Time Metrics Stream
**GET** `/v1/atlas/metrics/stream`

Server-Sent Events stream for real-time metrics:

```
Content-Type: text/event-stream

data: {"timestamp": 1704067200, "active_requests": 5, "avg_latency": 2.1}

data: {"timestamp": 1704067201, "active_requests": 3, "avg_latency": 2.2}
```

#### Memory Operations
**POST** `/v1/atlas/memory/{operation}`

Manage persistent memory contexts:

**Save Context:**
```json
// POST /v1/atlas/memory/save
{
  "context_id": "user123_conversation",
  "state": {
    "conversation_history": [...],
    "user_preferences": {...},
    "context_memory": {...}
  }
}
```

**Load Context:**
```json
// POST /v1/atlas/memory/load
{
  "context_id": "user123_conversation"
}
```

**List Contexts:**
```json
// POST /v1/atlas/memory/list
{}
```

**Delete Context:**
```json
// POST /v1/atlas/memory/delete
{
  "context_id": "user123_conversation"
}
```

## Configuration

### Environment Variables
```bash
# Enable ATLAS features
export ATLAS_ENABLED=1

# Configure memory persistence
export ATLAS_MEMORY_PATH="/path/to/atlas/memory"
export ATLAS_AUTO_SAVE_INTERVAL=300  # seconds

# Performance tuning
export ATLAS_MAX_CONCURRENT=32
export ATLAS_LATENCY_BUDGET_MS=5.0
export ATLAS_METRICS_INTERVAL=1000   # milliseconds
```

### Server Configuration
```cpp
// C++ configuration example
atlas::AtlasServerConfig config;
config.enable_memory_persistence = true;
config.memory_save_path = "./atlas_memory";
config.max_concurrent_atlas_requests = 32;
config.api_latency_budget_ms = 5.0f;
config.enable_metrics_collection = true;
config.metrics_update_interval_ms = 1000;
```

## Integration Guide

### Simple Integration
For basic ATLAS integration, use the convenience macros:

```cpp
#include "atlas-integration.hpp"

// Initialize ATLAS
atlas::init_atlas(model);

// Check for ATLAS requests
if (atlas::is_atlas_request(request)) {
    auto response = atlas::process_with_atlas(request);
    return atlas::enhance_response_with_atlas(response, true);
}

// Get ATLAS information
auto info = atlas::get_atlas_info();
```

### Advanced Integration
For full control, use the ATLAS server directly:

```cpp
#include "atlas-server.hpp"

// Configure and initialize
atlas::AtlasServerConfig config;
config.enable_memory_persistence = true;
config.max_concurrent_atlas_requests = 16;

atlas::AtlasServer atlas_server(config);
atlas_server.initialize(llama_model);

// Process requests
auto response = atlas_server.handle_completion_request(request);

// Get metrics
auto metrics = atlas_server.handle_metrics_request();
```

### HTTP Server Integration
Add ATLAS endpoints to your HTTP server:

```cpp
#include "atlas-endpoints.hpp"

// Add endpoints using the macro
ATLAS_ADD_ENDPOINTS(server);

// Or manually register endpoints
server.Post("/v1/atlas/config", handle_atlas_config);
server.Get("/v1/atlas/metrics", handle_atlas_metrics);
server.Post("/v1/atlas/memory/save", handle_atlas_memory_save);
```

## Performance Optimization

### Context Pool Tuning
```cpp
// Optimize for your workload
config.max_concurrent_atlas_requests = std::thread::hardware_concurrency();
config.context_pool_size = config.max_concurrent_atlas_requests * 2;
```

### Memory Management
```cpp
// Configure memory settings
config.l1_cache_size = 64 * 1024 * 1024;    // 64MB
config.l2_cache_size = 256 * 1024 * 1024;   // 256MB  
config.l3_cache_size = 1024 * 1024 * 1024;  // 1GB
```

### Metrics Optimization
```cpp
// Balance metrics detail vs performance
config.metrics_update_interval_ms = 5000;    // Less frequent updates
config.metrics_history_size = 500;           // Smaller history
```

## Error Handling

### Response Format
All ATLAS endpoints return structured error responses:

```json
{
  "error": {
    "code": 400,
    "message": "Validation error in field 'atlas.memory_layers': value must be between 1 and 10",
    "type": "validation_error",
    "timestamp": 1704067200
  }
}
```

### Common Error Codes
- **400**: Invalid request parameters
- **429**: Rate limit exceeded (too many concurrent requests)
- **503**: Service unavailable (context pool exhausted)
- **500**: Internal server error

### Error Recovery
ATLAS provides graceful degradation:
- Falls back to standard processing if ATLAS contexts are unavailable
- Continues operation with reduced functionality on component failures
- Automatically retries failed operations with backoff

## Monitoring and Observability

### Key Metrics to Monitor
1. **Request Latency**: Should stay below 5ms average
2. **Context Pool Utilization**: Monitor for bottlenecks
3. **Memory Usage**: Track persistence overhead
4. **Cache Hit Ratio**: Higher ratios indicate better performance
5. **Error Rates**: Monitor for service degradation

### Alerting Thresholds
```yaml
# Example alerting configuration
alerts:
  - metric: avg_latency_ms
    threshold: 5.0
    condition: greater_than
  - metric: context_pool_utilization
    threshold: 0.8
    condition: greater_than
  - metric: error_rate
    threshold: 0.05
    condition: greater_than
```

### Logging
ATLAS provides structured logging:
```
[ATLAS] Context pool initialized with 8 contexts
[ATLAS] Request processed in 2.3ms (memory_layers=3, cache_hit=true)
[ATLAS] Memory state saved for session: user123_conversation
[ATLAS] Metrics updated: 1250 requests, 2.1ms avg latency
```

## Troubleshooting

### Common Issues

**High Latency**
- Check context pool size vs concurrent requests
- Monitor memory usage and garbage collection
- Verify network and storage performance for persistence

**Context Pool Exhaustion**
```
Error: No available ATLAS contexts (503)
```
- Increase `max_concurrent_atlas_requests`
- Check for context leaks or long-running requests
- Monitor request processing times

**Memory Persistence Issues**
```
Error: Failed to save context state (500)
```
- Check disk space and permissions for `memory_save_path`
- Verify JSON serialization compatibility
- Monitor memory usage during persistence

**Build Errors**
```
Error: ATLAS headers not found
```
- Ensure `DLLAMA_ATLAS=ON` during CMake configuration
- Verify all ATLAS source files are present
- Check C++17 compiler support

### Debug Mode
Enable verbose ATLAS logging:
```cpp
config.verbose_logging = true;
config.log_level = atlas::LogLevel::DEBUG;
```

### Performance Profiling
```bash
# Profile with perf
perf record -g ./llama-server --model model.gguf
perf report

# Monitor with htop/top
htop -p $(pgrep llama-server)

# Memory profiling with valgrind
valgrind --tool=massif ./llama-server --model model.gguf
```

## Best Practices

### Session Management
- Use meaningful session IDs for persistence
- Implement session cleanup for inactive users
- Consider session expiration policies

### Caching Strategy
- Use "adaptive" for mixed workloads
- Use "lru" for temporal locality patterns  
- Use "lfu" for frequency-based patterns

### Resource Management
- Size context pool based on expected concurrent users
- Monitor memory usage and implement limits
- Use appropriate batch sizes for throughput

### Security Considerations
- Validate all session IDs and context IDs
- Implement rate limiting per user/session
- Secure memory persistence storage
- Monitor for resource exhaustion attacks

## Migration Guide

### From Standard llama-server
1. Rebuild with `DLLAMA_ATLAS=ON`
2. Update client code to include optional `atlas` parameters
3. Configure memory persistence paths
4. Set up monitoring for new metrics

### API Compatibility
ATLAS maintains full backward compatibility:
- Standard OpenAI requests work without modification
- ATLAS features are opt-in via request parameters
- Existing clients continue working unchanged

## Support and Contributing

### Getting Help
- Check the troubleshooting section above
- Review error logs and metrics
- Test with minimal configurations
- Profile performance bottlenecks

### Contributing
- Follow existing code style and patterns
- Add comprehensive tests for new features
- Update documentation for API changes
- Benchmark performance impact

### Reporting Issues
Include in bug reports:
- ATLAS configuration used
- Request examples that fail
- Error logs and stack traces
- Performance metrics during issues
- System specifications and versions

## License and Attribution

ATLAS integration follows the same license as llama.cpp. See the main project LICENSE file for details.

---

For more information, see the [ATLAS Test Suite Documentation](../../tests/README_ATLAS_TESTS.md) and the main [llama.cpp documentation](https://github.com/ggml-org/llama.cpp).