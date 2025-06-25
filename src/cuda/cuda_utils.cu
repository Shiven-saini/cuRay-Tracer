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
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
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

void checkKernelLaunch(const char* kernelName) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error (" << kernelName << "): " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution error (" << kernelName << "): " 
                  << cudaGetErrorString(err) << std::endl;
    }
}

} // namespace CudaUtils