#include <iostream>
#include <cuda_runtime_api.h>
#include "../include/device_management.hpp"
#include "../include/utils.h"

void ensure_cuda_device_available() {
    int deviceCount;
    cudaError error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        throw std::runtime_error("Can't find cuda devices");
    }
}

void set_cuda_device(int device) {
    ensure_cuda_device_available();
    CUDA_CALL(cudaSetDevice(device));
    int currentDevice;
    CUDA_CALL(cudaGetDevice(&currentDevice));
    cudaDeviceProp deviceProp = cudaDevicePropDontCare;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, currentDevice));
    std::cout << "Using Cuda device: " << deviceProp.name << std::endl;
}


