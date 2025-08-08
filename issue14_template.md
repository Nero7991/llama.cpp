    def test_cpu_only_deployment(self):
        """Test ATLAS server on CPU-only environment"""
        container = self.docker_client.containers.run(
            "atlas-server-cpu:latest",
            command=["./build/bin/llama-server", "-m", "test-model.gguf", "--atlas", "--port", "8080"],
            ports={"8080/tcp": 8080},
            detach=True,
            remove=True
        )
        
        try:
            # Wait for startup
            time.sleep(15)
            
            # Test basic functionality
            response = requests.get("http://localhost:8080/v1/atlas/status")
            assert response.status_code == 200
            
            # Test completion
            completion_response = requests.post(
                "http://localhost:8080/v1/completions",
                json={
                    "prompt": "Test prompt",
                    "max_tokens": 10,
                    "atlas": {"enabled": True}
                }
            )
            assert completion_response.status_code == 200
            
        finally:
            container.stop()
    
    def test_gpu_deployment(self):
        """Test ATLAS server with GPU acceleration"""
        container = self.docker_client.containers.run(
            "atlas-server-gpu:latest",
            command=["./build/bin/llama-server", "-m", "test-model.gguf", "--atlas", "--cuda", "--port", "8080"],
            ports={"8080/tcp": 8080},
            runtime="nvidia",
            detach=True,
            remove=True
        )
        
        try:
            time.sleep(15)
            
            # Verify GPU utilization
            status_response = requests.get("http://localhost:8080/v1/atlas/status")
            status_data = status_response.json()
            
            # Should have better performance metrics with GPU
            assert "performance" in status_data
            assert status_data["performance"]["tokens_per_second"] > 10
            
        finally:
            container.stop()
```

#### Kubernetes Deployment Tests
```yaml
# tests/k8s/atlas-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: atlas-server-test
spec:
  replicas: 2
  selector:
    matchLabels:
      app: atlas-server-test
  template:
    metadata:
      labels:
        app: atlas-server-test
    spec:
      containers:
      - name: atlas-server
        image: atlas-server:latest
        command: ["./build/bin/llama-server"]
        args: ["-m", "/models/test-model.gguf", "--atlas", "--port", "8080"]
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi" 
            cpu: "4"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: atlas-memory
          mountPath: /atlas-memory
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: atlas-memory
        persistentVolumeClaim:
          claimName: atlas-memory-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: atlas-server-service
spec:
  selector:
    app: atlas-server-test
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### 6. Error Handling and Recovery Tests

#### Fault Injection Testing
```python
class AtlasFaultInjectionTests:
    def test_memory_file_corruption(self):
        """Test recovery from corrupted memory files"""
        # Create and save valid memory
        save_response = requests.post(
            "http://localhost:8080/v1/atlas/memory/save",
            json={"filename": "test_corruption.atlas"}
        )
        assert save_response.status_code == 200
        
        # Corrupt the memory file
        with open("test_corruption.atlas", "r+b") as f:
            f.seek(100)
            f.write(b"CORRUPTED_DATA")
        
        # Attempt to load corrupted file
        load_response = requests.post(
            "http://localhost:8080/v1/atlas/memory/load",
            json={"filename": "test_corruption.atlas"}
        )
        
        # Should handle gracefully
        assert load_response.status_code == 400
        error_data = load_response.json()
        assert "error" in error_data
        assert "corruption" in error_data["error"]["message"].lower()
    
    def test_disk_space_exhaustion(self):
        """Test behavior when disk space is exhausted"""
        # This would require actual disk space manipulation
        # Implementation depends on test environment
        pass
    
    def test_memory_exhaustion(self):
        """Test behavior under memory pressure"""
        # Create many concurrent requests to stress memory
        concurrent_requests = []
        for i in range(50):
            request_data = {
                "prompt": "Generate a very long response..." * 100,
                "max_tokens": 2048,
                "atlas": {
                    "enabled": True,
                    "window_size": 2048
                }
            }
            concurrent_requests.append(request_data)
        
        # Send all requests simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(
                    requests.post,
                    "http://localhost:8080/v1/completions",
                    json=req
                )
                for req in concurrent_requests
            ]
            
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Should handle gracefully without crashes
        successful_requests = [r for r in results if r.status_code == 200]
        failed_requests = [r for r in results if r.status_code != 200]
        
        # Should have some successful requests
        assert len(successful_requests) > 0
        
        # Failed requests should have appropriate error messages
        for failed_req in failed_requests:
            if failed_req.status_code == 503:  # Service unavailable
                error_data = failed_req.json()
                assert "memory" in error_data["error"]["message"].lower()
```

## Testing Requirements

### API Functionality Tests
- [ ] **All endpoints**: Every ATLAS API endpoint functions correctly
- [ ] **Parameter validation**: Invalid parameters return appropriate error codes
- [ ] **OpenAI compatibility**: Standard OpenAI clients work with ATLAS extensions
- [ ] **Authentication**: API key handling (if implemented)
- [ ] **Rate limiting**: Rate limiting works correctly with ATLAS requests

### Performance and Load Tests
- [ ] **Concurrent requests**: Handle 50+ concurrent ATLAS-enhanced requests
- [ ] **Memory persistence load**: 1000+ save/load cycles without issues
- [ ] **Long-running stability**: 24+ hour continuous operation
- [ ] **Resource usage**: Memory and CPU usage within acceptable bounds
- [ ] **Context length scaling**: Performance scales linearly with context length

### Integration and Compatibility Tests
- [ ] **Client libraries**: Works with OpenAI Python, Node.js, Go clients
- [ ] **LangChain integration**: Compatible with LangChain LLM interface
- [ ] **Curl commands**: All functionality accessible via curl
- [ ] **Docker deployment**: Works in containerized environments
- [ ] **Kubernetes deployment**: Scales properly in K8s clusters

### Error Handling and Recovery Tests
- [ ] **Graceful degradation**: ATLAS failures don't crash server
- [ ] **Memory file corruption**: Handles corrupted memory files gracefully
- [ ] **Disk space issues**: Appropriate errors when disk full
- [ ] **Network interruptions**: Handles client disconnections properly
- [ ] **Resource exhaustion**: Proper handling of memory/CPU limits

## Implementation Files

### Test Framework Core
- `tests/atlas/server/framework.py` - Core testing framework
- `tests/atlas/server/api_tests.py` - API endpoint tests
- `tests/atlas/server/load_tests.py` - Load and performance tests
- `tests/atlas/server/integration_tests.py` - Integration tests

### Performance Testing
- `tests/atlas/server/performance_benchmark.cpp` - C++ performance benchmarks
- `tests/atlas/server/memory_profiler.cpp` - Memory usage profiling
- `tests/atlas/server/concurrent_test.py` - Concurrency testing

### Deployment Testing
- `tests/atlas/deployment/docker/` - Docker deployment tests
- `tests/atlas/deployment/k8s/` - Kubernetes deployment tests
- `tests/atlas/deployment/multi_env.py` - Multi-environment testing

### Automation Scripts
- `scripts/run-atlas-server-tests.sh` - Automated test execution
- `scripts/atlas-load-test.py` - Load testing script
- `scripts/atlas-monitor.py` - Real-time monitoring script

### CI/CD Integration
- `.github/workflows/atlas-server-tests.yml` - GitHub Actions workflow
- `tests/atlas/server/ci_config.py` - CI configuration helpers

## Success Criteria

### Functional Requirements
- [ ] 100% API endpoint test coverage with passing tests
- [ ] All OpenAI-compatible clients work with ATLAS extensions
- [ ] Memory persistence works reliably across all scenarios
- [ ] Error handling covers all identified edge cases
- [ ] Performance meets documented benchmarks

### Performance Requirements
- [ ] Handle 50+ concurrent requests without degradation
- [ ] API latency overhead <10ms compared to standard inference
- [ ] Memory save/load operations complete within documented timeframes
- [ ] 24+ hour stability testing passes without issues
- [ ] Resource usage stays within documented bounds

### Quality Requirements
- [ ] Zero crashes during stress testing
- [ ] All error conditions return appropriate HTTP status codes
- [ ] Complete test documentation with runbook
- [ ] Automated CI/CD pipeline catches regressions
- [ ] Production deployment guide verified through testing

## Automated Test Execution

### Local Development Testing
```bash
# Quick smoke test
make test-atlas-server-smoke

# Full test suite (30 minutes)
make test-atlas-server-full

# Performance benchmarks (60 minutes)  
make test-atlas-server-performance

# Load testing (configurable duration)
./scripts/atlas-load-test.py --duration 300 --concurrent 20
```

### CI/CD Integration
```bash
# Trigger full test suite in CI
git push origin feature/atlas-enhancement

# Manual CI trigger for specific tests
gh workflow run atlas-server-tests.yml \
  --ref feature/atlas-enhancement \
  -f test_type=load_test \
  -f duration=600
```

## Performance Benchmarks

### Expected Performance Targets
- **API Latency**: <50ms for simple completions with ATLAS
- **Throughput**: >20 requests/second for 512-token responses
- **Memory Usage**: <8GB total for 7B model with ATLAS
- **Context Scaling**: Linear performance up to 32K context
- **Concurrent Users**: Support 50+ simultaneous users

### Monitoring Metrics
- Request latency (p50, p95, p99)
- Throughput (requests/second, tokens/second)
- Memory usage (total, ATLAS-specific)
- CPU utilization
- GPU utilization (if applicable)
- Error rates and types

## Dependencies
- Issues #11-13: ATLAS API integration, memory persistence, direct GGUF support
- Python testing frameworks (pytest, requests, aiohttp)
- Load testing tools
- Docker and Kubernetes for deployment testing
- Monitoring and profiling tools

## Estimated Effort
**2-3 weeks** for comprehensive testing framework implementation

## References
- llama-server documentation
- OpenAI API specification for compatibility testing
- Load testing best practices
- Production deployment testing methodologies
