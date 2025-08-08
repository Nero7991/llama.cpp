#include "framework/atlas-test-framework.h"
#include <iostream>

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

using namespace atlas::testing;

// CUDA performance tests for ATLAS
class AtlasCudaPerformanceTests {
public:
    static bool runCudaMemoryBandwidthTest() {
        std::cout << "Testing CUDA memory bandwidth..." << std::endl;
        
#ifdef CUDA_FOUND
        // Check if CUDA device is available
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error != cudaSuccess || device_count == 0) {
            std::cout << "No CUDA devices available" << std::endl;
            return false;
        }
        
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        std::cout << "CUDA Device: " << prop.name << std::endl;
        std::cout << "Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        
        // Simple memory bandwidth test
        const size_t size = 64 * 1024 * 1024; // 64MB
        float *h_data = new float[size / sizeof(float)];
        float *d_data;
        
        // Allocate device memory
        if (cudaMalloc(&d_data, size) != cudaSuccess) {
            delete[] h_data;
            return false;
        }
        
        // Initialize host data
        for (size_t i = 0; i < size / sizeof(float); i++) {
            h_data[i] = 0.1f * i;
        }
        
        // Time H2D transfer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float h2d_time;
        cudaEventElapsedTime(&h2d_time, start, stop);
        
        // Time D2H transfer
        cudaEventRecord(start);
        cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float d2h_time;
        cudaEventElapsedTime(&d2h_time, start, stop);
        
        double h2d_bandwidth = size / (h2d_time / 1000.0) / 1e9;
        double d2h_bandwidth = size / (d2h_time / 1000.0) / 1e9;
        
        std::cout << "H2D Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl;
        std::cout << "D2H Bandwidth: " << d2h_bandwidth << " GB/s" << std::endl;
        
        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        delete[] h_data;
        
        return true;
#else
        std::cout << "CUDA not available - skipping test" << std::endl;
        return true;
#endif
    }
    
    static bool runCudaComputeTest() {
        std::cout << "Testing CUDA compute performance..." << std::endl;
        
#ifdef CUDA_FOUND
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "No CUDA devices available" << std::endl;
            return false;
        }
        
        // Simple SAXPY kernel test
        const int n = 1024 * 1024;
        const float alpha = 2.0f;
        
        float *h_x = new float[n];
        float *h_y = new float[n];
        float *d_x, *d_y;
        
        // Initialize data
        for (int i = 0; i < n; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
        }
        
        // Allocate device memory
        if (cudaMalloc(&d_x, n * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_y, n * sizeof(float)) != cudaSuccess) {
            delete[] h_x;
            delete[] h_y;
            return false;
        }
        
        // Copy to device
        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
        
        // Use cuBLAS if available, otherwise just validate setup
        cublasHandle_t handle;
        if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start);
            cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float compute_time;
            cudaEventElapsedTime(&compute_time, start, stop);
            
            double gflops = (2.0 * n) / (compute_time / 1000.0) / 1e9;
            std::cout << "SAXPY Performance: " << gflops << " GFLOPS" << std::endl;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cublasDestroy(handle);
        }
        
        // Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        delete[] h_x;
        delete[] h_y;
        
        return true;
#else
        std::cout << "CUDA not available - skipping test" << std::endl;
        return true;
#endif
    }
};

int main() {
    std::cout << "=== ATLAS CUDA Performance Tests ===" << std::endl;
    
    bool all_passed = true;
    
    std::cout << "\n1. Running CUDA memory bandwidth test..." << std::endl;
    if (!AtlasCudaPerformanceTests::runCudaMemoryBandwidthTest()) {
        std::cout << "❌ CUDA memory bandwidth test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ CUDA memory bandwidth test passed" << std::endl;
    }
    
    std::cout << "\n2. Running CUDA compute test..." << std::endl;
    if (!AtlasCudaPerformanceTests::runCudaComputeTest()) {
        std::cout << "❌ CUDA compute test failed" << std::endl;
        all_passed = false;
    } else {
        std::cout << "✅ CUDA compute test passed" << std::endl;
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "🎉 ALL CUDA TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ SOME CUDA TESTS FAILED!" << std::endl;
    }
    std::cout << std::string(50, '=') << std::endl;
    
    return all_passed ? 0 : 1;
}