# ATLAS API Reference

## Overview

ATLAS (Advanced Tensor Learning and Attention System) extends the llama-server with enhanced capabilities while maintaining full OpenAI API compatibility. All standard OpenAI endpoints work unchanged, with optional ATLAS parameters for enhanced functionality.

## Base URL
```
http://localhost:8080  # Default server address
```

## Authentication
ATLAS follows the same authentication model as llama-server. If API keys are configured, include them in requests:
```bash
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     ...
```

## Enhanced OpenAI Endpoints

### Chat Completions
**Endpoint:** `POST /v1/chat/completions`

Standard OpenAI Chat Completions API with optional ATLAS extensions.

#### Request Format
```json
{
  "model": "your-model-name",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "temperature": 0.7,
  "max_tokens": 150,
  "stream": false,
  
  "atlas": {
    "memory_layers": 3,
    "cache_strategy": "adaptive",
    "session_id": "user123",
    "batch_size": 8
  }
}
```

#### ATLAS Parameters
| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `memory_layers` | integer | 1-10 | 3 | Number of memory layers to utilize for enhanced context |
| `cache_strategy` | string | "lru", "lfu", "adaptive" | "adaptive" | Caching strategy for optimal performance |
| `session_id` | string | max 128 chars | null | Persistent session identifier for memory continuity |
| `batch_size` | integer | 1-32 | 8 | Batch processing size for efficiency |

#### Response Format
Standard OpenAI response with ATLAS metadata:
```json
{
  "id": "atlas_chatcmpl_123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "your-model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 15,
    "total_tokens": 35
  },
  "atlas": {
    "processed": true,
    "version": "6A",
    "context_id": "atlas_ctx_2",
    "cache_hits": 3,
    "memory_layers_used": 3,
    "performance_ms": 2.3,
    "timestamp": 1704067200
  }
}
```

#### Example Usage
```bash
# Basic chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ]
  }'

# ATLAS-enhanced chat completion
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b",
    "messages": [
      {"role": "user", "content": "Continue our previous conversation about ML"}
    ],
    "atlas": {
      "memory_layers": 5,
      "session_id": "user123_ml_discussion",
      "cache_strategy": "adaptive"
    }
  }'
```

### Text Completions
**Endpoint:** `POST /v1/completions`

Standard OpenAI Completions API with ATLAS enhancements.

#### Request Format
```json
{
  "model": "your-model-name",
  "prompt": "The capital of France is",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 1.0,
  "n": 1,
  "stream": false,
  "stop": ["\n"],
  
  "atlas": {
    "memory_layers": 2,
    "session_id": "completion_session",
    "cache_strategy": "lru"
  }
}
```

#### Response Format
```json
{
  "id": "atlas_cmpl_456",
  "object": "text_completion",
  "created": 1704067200,
  "model": "your-model-name",
  "choices": [
    {
      "text": "Paris, the beautiful capital city of France.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 9,
    "total_tokens": 15
  },
  "atlas": {
    "processed": true,
    "context_id": "atlas_ctx_1",
    "cache_hits": 1,
    "memory_layers_used": 2,
    "performance_ms": 1.8
  }
}
```

## ATLAS-Specific Endpoints

### System Status
**Endpoint:** `GET /v1/atlas/status`

Returns current ATLAS system status and health information.

#### Response
```json
{
  "status": "success",
  "data": {
    "healthy": true,
    "initialized": true,
    "context_pool_size": 8,
    "metrics_enabled": true,
    "memory_persistence": true,
    "uptime_seconds": 3600,
    "atlas_enabled": true,
    "atlas_version": "6A"
  },
  "timestamp": 1704067200
}
```

#### Status Fields
| Field | Type | Description |
|-------|------|-------------|
| `healthy` | boolean | Overall system health status |
| `initialized` | boolean | ATLAS initialization status |
| `context_pool_size` | integer | Number of available contexts |
| `metrics_enabled` | boolean | Metrics collection status |
| `memory_persistence` | boolean | Memory persistence status |
| `uptime_seconds` | integer | System uptime in seconds |

### Performance Metrics
**Endpoint:** `GET /v1/atlas/metrics`

Provides comprehensive real-time performance metrics.

#### Query Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | string | Response format: "json" (default), "prometheus" |
| `window` | integer | Time window in seconds for metrics (default: 60) |

#### Response
```json
{
  "status": "success",
  "data": {
    "current": {
      "total_requests": 1250,
      "active_requests": 3,
      "completed_requests": 1247,
      "failed_requests": 3,
      "success_rate": 0.998,
      
      "latency": {
        "avg_ms": 2.3,
        "median_ms": 2.1,
        "p95_ms": 4.8,
        "p99_ms": 8.2,
        "min_ms": 0.8,
        "max_ms": 15.2
      },
      
      "throughput": {
        "requests_per_second": 125.5,
        "tokens_per_second": 2250.8,
        "avg_rps_1min": 120.3
      },
      
      "cache": {
        "hits": 890,
        "misses": 357,
        "hit_ratio": 0.714,
        "avg_hit_ratio": 0.695
      },
      
      "memory": {
        "current_bytes": 524288000,
        "peak_bytes": 1073741824,
        "avg_bytes": 419430400,
        "usage_percent": 65.2
      },
      
      "system": {
        "cpu_usage_percent": 45.8,
        "memory_usage_percent": 72.1
      },
      
      "errors": {
        "total_types": 3,
        "recent_count": 5,
        "by_type": {
          "timeout": 2,
          "validation_error": 2,
          "internal_error": 1
        }
      }
    },
    "history_size": 1000,
    "server_health": true
  },
  "timestamp": 1704067200
}
```

### Real-Time Metrics Stream
**Endpoint:** `GET /v1/atlas/metrics/stream`

Server-Sent Events stream for real-time metrics monitoring.

#### Response Headers
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Access-Control-Allow-Origin: *
```

#### Stream Format
```
data: {"timestamp": 1704067200, "active_requests": 5, "avg_latency": 2.1, "rps": 125.5}

data: {"timestamp": 1704067201, "active_requests": 3, "avg_latency": 2.2, "rps": 128.1}

data: {"timestamp": 1704067202, "active_requests": 7, "avg_latency": 2.0, "rps": 130.2}
```

#### JavaScript Client Example
```javascript
const eventSource = new EventSource('http://localhost:8080/v1/atlas/metrics/stream');

eventSource.onmessage = function(event) {
  const metrics = JSON.parse(event.data);
  console.log('Current metrics:', metrics);
  
  // Update your dashboard
  updateLatencyChart(metrics.avg_latency);
  updateThroughputChart(metrics.rps);
  updateActiveRequests(metrics.active_requests);
};

eventSource.onerror = function(event) {
  console.error('Metrics stream error:', event);
};
```

### Memory Management

#### Save Context
**Endpoint:** `POST /v1/atlas/memory/save`

Saves conversation context for future sessions.

##### Request
```json
{
  "context_id": "user123_conversation",
  "state": {
    "conversation_history": [
      {"role": "user", "content": "What is AI?"},
      {"role": "assistant", "content": "AI is..."}
    ],
    "user_preferences": {
      "response_style": "detailed",
      "topic_interests": ["technology", "science"]
    },
    "context_memory": {
      "last_topic": "artificial_intelligence",
      "conversation_length": 15
    }
  }
}
```

##### Response
```json
{
  "status": "success",
  "operation": "save",
  "context_id": "user123_conversation",
  "message": "Context saved successfully",
  "timestamp": 1704067200
}
```

#### Load Context
**Endpoint:** `POST /v1/atlas/memory/load`

Loads previously saved conversation context.

##### Request
```json
{
  "context_id": "user123_conversation"
}
```

##### Response
```json
{
  "status": "success",
  "operation": "load",
  "context_id": "user123_conversation",
  "data": {
    "conversation_history": [...],
    "user_preferences": {...},
    "context_memory": {...}
  },
  "message": "Context loaded successfully"
}
```

#### List Contexts
**Endpoint:** `POST /v1/atlas/memory/list`

Lists all saved contexts.

##### Request
```json
{}
```

##### Response
```json
{
  "status": "success",
  "operation": "list",
  "contexts": [
    "user123_conversation",
    "user456_chat",
    "session_789"
  ]
}
```

#### Delete Context
**Endpoint:** `POST /v1/atlas/memory/delete`

Deletes a saved context.

##### Request
```json
{
  "context_id": "user123_conversation"
}
```

##### Response
```json
{
  "status": "success",
  "operation": "delete",
  "context_id": "user123_conversation",
  "message": "Context deleted successfully"
}
```

## Error Responses

### Error Format
All errors follow a consistent format:

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

### Error Types

#### 400 - Bad Request
**Validation Errors:**
```json
{
  "error": {
    "code": 400,
    "message": "Invalid JSON in request body",
    "type": "validation_error"
  }
}
```

**Parameter Errors:**
```json
{
  "error": {
    "code": 400,
    "message": "Field 'atlas.memory_layers' must be between 1 and 10",
    "type": "validation_error"
  }
}
```

#### 429 - Rate Limited
```json
{
  "error": {
    "code": 429,
    "message": "Too many requests. Please try again later.",
    "type": "rate_limit_exceeded",
    "retry_after": 5
  }
}
```

#### 503 - Service Unavailable
```json
{
  "error": {
    "code": 503,
    "message": "No available ATLAS contexts. Please try again later.",
    "type": "resource_exhausted"
  }
}
```

#### 500 - Internal Server Error
```json
{
  "error": {
    "code": 500,
    "message": "Internal server error: Context processing failed",
    "type": "internal_error"
  }
}
```

## Rate Limiting

ATLAS implements intelligent rate limiting to maintain performance:

### Limits
- **Concurrent Requests:** 32 simultaneous requests (configurable)
- **Per-Session Rate:** 100 requests per minute per session_id
- **Global Rate:** 1000 requests per minute per API key

### Headers
Rate limit information is included in response headers:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1704067260
X-ATLAS-Processed: true
```

## Client SDKs and Examples

### Python Client
```python
import requests

class AtlasClient:
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def chat_completion(self, messages, session_id=None, **kwargs):
        data = {
            "model": "your-model",
            "messages": messages,
            **kwargs
        }
        
        if session_id:
            data["atlas"] = {
                "session_id": session_id,
                "memory_layers": 3,
                "cache_strategy": "adaptive"
            }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def get_metrics(self):
        response = requests.get(f"{self.base_url}/v1/atlas/metrics")
        return response.json()

# Usage example
client = AtlasClient()
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello!"}],
    session_id="user123"
)
print(response)
```

### JavaScript/Node.js Client
```javascript
class AtlasClient {
    constructor(baseUrl = 'http://localhost:8080', apiKey = null) {
        this.baseUrl = baseUrl;
        this.headers = { 'Content-Type': 'application/json' };
        if (apiKey) {
            this.headers.Authorization = `Bearer ${apiKey}`;
        }
    }
    
    async chatCompletion(messages, sessionId = null, options = {}) {
        const data = {
            model: 'your-model',
            messages,
            ...options
        };
        
        if (sessionId) {
            data.atlas = {
                session_id: sessionId,
                memory_layers: 3,
                cache_strategy: 'adaptive'
            };
        }
        
        const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(data)
        });
        
        return response.json();
    }
    
    async getMetrics() {
        const response = await fetch(`${this.baseUrl}/v1/atlas/metrics`);
        return response.json();
    }
    
    createMetricsStream() {
        return new EventSource(`${this.baseUrl}/v1/atlas/metrics/stream`);
    }
}

// Usage example
const client = new AtlasClient();
const response = await client.chatCompletion(
    [{ role: 'user', content: 'Hello!' }],
    'user123'
);
console.log(response);
```

### cURL Examples

#### Basic Chat Completion
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "max_tokens": 200
  }'
```

#### ATLAS-Enhanced Chat with Session
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": "Continue our quantum computing discussion"}
    ],
    "atlas": {
      "session_id": "quantum_learning_session",
      "memory_layers": 5,
      "cache_strategy": "adaptive"
    }
  }'
```

#### Get System Metrics
```bash
curl http://localhost:8080/v1/atlas/metrics | jq '.'
```

#### Save Conversation Context
```bash
curl -X POST http://localhost:8080/v1/atlas/memory/save \
  -H "Content-Type: application/json" \
  -d '{
    "context_id": "quantum_discussion",
    "state": {
      "topic": "quantum_computing",
      "expertise_level": "beginner",
      "previous_questions": ["what is quantum computing", "quantum vs classical"]
    }
  }'
```

## Performance Guidelines

### Optimal Configuration
- Use `memory_layers: 3-5` for balanced performance and context
- Choose `cache_strategy: "adaptive"` for mixed workloads
- Keep `session_id` consistent for conversation continuity
- Set appropriate `max_tokens` to control response time

### Monitoring Performance
- Monitor `/v1/atlas/metrics` for latency trends
- Watch for context pool saturation (active_requests near max)
- Track cache hit ratios for optimization opportunities
- Set up alerts for error rate increases

### Best Practices
1. **Session Management:** Use meaningful session IDs
2. **Memory Layers:** Start with 3, increase for complex conversations
3. **Batch Processing:** Use appropriate batch sizes for throughput
4. **Error Handling:** Implement retry logic with exponential backoff
5. **Monitoring:** Set up real-time metrics dashboards

---

For implementation details, see the [ATLAS Integration Guide](README_ATLAS.md) and [Test Suite Documentation](../../tests/README_ATLAS_TESTS.md).