#include "cuda/cuda_utils.h"
#include <iostream>

namespace CudaUtils {

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s):" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

bool checkCudaCapabilities() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Check for minimum compute capability (3.0)
    if (prop.major < 3) {
        std::cerr << "CUDA compute capability 3.0 or higher required!" << std::endl;
        return false;
    }
    
    return true;
}

} // namespace CudaUtils