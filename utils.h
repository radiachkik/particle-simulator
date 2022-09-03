#ifndef LIFE_SIMULATOR_UTILS_H
#define LIFE_SIMULATOR_UTILS_H

#include <string>

#define CUDA_CALL(x) \
    do { \
        if((x) != cudaSuccess) { \
            printf("Error at %s:%d\n",__FILE__,__LINE__); \
            throw std::runtime_error("Cuda Error: " + (std::string)cudaGetErrorString(x)); \
        } \
    } while(0)

#endif //LIFE_SIMULATOR_UTILS_H
