#include "simulation.hpp"
#include "utils.h"
#include <curand_kernel.h>
#include <iostream>

namespace simulation {
    namespace {
        __constant__ float *dev_coordinates;
        __constant__ float *dev_velocities;
        __constant__ float *dev_gravities;
        __constant__ float *dev_norm_coordinates;
        __constant__ float *dev_forces;
        __constant__ unsigned int dev_num_point_clouds;
        __constant__ const unsigned int dev_dimensions = 2;
        __constant__ unsigned int dev_points_per_cloud;
        __constant__ float *dev_distance_thresholds;
        __constant__ bool dev_border_enabled;

        const unsigned int host_dimensions = 2;
        simulationConfig *config;
        float *host_norm_coordinates;

        __global__
        void generate_random_kernel(curandState *rand_state)
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;

            unsigned int total_values = dev_num_point_clouds * dev_points_per_cloud * dev_dimensions;

            curand_init(134, id, 0, &rand_state[id]);
            curandState localState = rand_state[id];
            for(unsigned int i = id; i < total_values; i += total_threads) {
                dev_coordinates[i] = curand_uniform(&localState) * 900 - 450;
            }
        }

        __global__
        void normalize_coordinates_kernel()
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;
            unsigned int total_values = dev_points_per_cloud * dev_dimensions * dev_num_point_clouds;

            for(unsigned int i = id; i < total_values; i += total_threads) {
                dev_norm_coordinates[i] = dev_coordinates[i] / 500;
            }
        }

        template <unsigned int blockSize>
        __global__
        void calculate_forces_kernel()
        {
            __shared__ float shared_force[blockSize * dev_dimensions];
            __shared__ float p1[dev_dimensions];

            // Each block is responsible for calculating the summed forces applied on p1
            for (auto p1_index = blockIdx.x; p1_index < dev_points_per_cloud * dev_num_point_clouds; p1_index += gridDim.x) {
                if (threadIdx.x < dev_dimensions) {
                    p1[threadIdx.x] = dev_coordinates[p1_index * dev_dimensions + threadIdx.x];
                }
                auto pc1_index = p1_index / dev_points_per_cloud;
                float force[dev_dimensions];
                for (float & d : force) {
                    d = 0.0f;
                }
                // Each thread is responsible for calculating the summed forces applied on p1 for some p2
                for(auto p2_index = threadIdx.x; p2_index < dev_points_per_cloud * dev_num_point_clouds; p2_index += blockDim.x) {
                    float *p2 = dev_coordinates + p2_index * dev_dimensions;
                    auto pc2_index = p2_index / dev_points_per_cloud;
                    float gravity = dev_gravities[pc1_index * dev_num_point_clouds + pc2_index];
                    float distance_threshold = dev_distance_thresholds[pc1_index * dev_num_point_clouds + pc2_index];

                    float distance[dev_dimensions];
                    for (int d = 0; d < dev_dimensions; d++) {
                        distance[d] = p2[d] - p1[d];
                    }
                    float r = normf((int)dev_dimensions, distance);
                    if (r < distance_threshold && r >= 1) {
                        for (int d = 0; d < dev_dimensions; d++) {
                            force[d] += (distance[d] / (r * 100) ) * gravity;
                        }
                    }
                }
                for (int d = 0; d < dev_dimensions; d++) {
                    shared_force[threadIdx.x * dev_dimensions + d] = force[d];
                }

                // Sum up forces in shared memory
                __syncthreads();
                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        for (unsigned int d = 0; d < dev_dimensions; d++) {
                            shared_force[(threadIdx.x * dev_dimensions) + d] += shared_force[(threadIdx.x + s) * dev_dimensions + d];
                        }
                    }
                    __syncthreads();
                }

                // Copy forces to global memory
                if (threadIdx.x < dev_dimensions) {
                    dev_forces[p1_index * dev_dimensions + threadIdx.x] = shared_force[threadIdx.x];
                }
            }
        }

        __global__
        void calculate_new_coordinates_kernel()
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;
            unsigned int total_values = dev_points_per_cloud * dev_dimensions * dev_num_point_clouds;

            for(unsigned int i = id; i < total_values; i += total_threads) {
                float new_velocity = (dev_velocities[i] + dev_forces[i]) * 0.5f;
                float new_coordinate = dev_coordinates[i] + new_velocity;
                dev_velocities[i] = new_velocity;
                dev_coordinates[i] = new_coordinate;
                if (dev_border_enabled) {
                    if (new_coordinate < -450) {
                        dev_coordinates[i] = -450;
                        dev_velocities[i] *= -1;
                    }else if (new_coordinate > 450) {
                        dev_coordinates[i] = 450;
                        dev_velocities[i] *= -1;
                    }
                }
            }
        }
    }

    void initialize_simulation(simulationConfig *simulation_config) {
        config = simulation_config;
        const unsigned int num_values = config->num_point_clouds * config->points_per_cloud * host_dimensions;

        host_norm_coordinates = (float*) malloc(num_values * sizeof(float));
        float *tmp_pointer;

        CUDA_CALL(cudaMemcpyToSymbol(dev_num_point_clouds, &config->num_point_clouds, sizeof(dev_num_point_clouds)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_points_per_cloud, &config->points_per_cloud, sizeof(dev_num_point_clouds)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_border_enabled, &config->border_enabled, sizeof(dev_border_enabled)));

        CUDA_CALL(cudaMalloc(&tmp_pointer, config->num_point_clouds * config->num_point_clouds * sizeof(float)));
        CUDA_CALL(cudaMemcpy(tmp_pointer, config->distance_thresholds, config->num_point_clouds * config->num_point_clouds * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(dev_distance_thresholds, &tmp_pointer, sizeof(dev_distance_thresholds)));

        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, config->num_point_clouds * config->num_point_clouds * sizeof(float)));
        CUDA_CALL(cudaMemcpy(tmp_pointer, config->gravities, config->num_point_clouds * config->num_point_clouds * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(dev_gravities, &tmp_pointer, sizeof(dev_gravities)));

        CUDA_CALL(cudaMalloc(&tmp_pointer, num_values * sizeof(float)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_values * sizeof(float)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_coordinates, &tmp_pointer, sizeof(dev_coordinates)));

        CUDA_CALL(cudaMalloc(&tmp_pointer, num_values * sizeof(float)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_values * sizeof(float)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_forces, &tmp_pointer, sizeof(dev_forces)));

        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, num_values * sizeof(float)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_values * sizeof(float)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_norm_coordinates, &tmp_pointer, sizeof(dev_norm_coordinates)));

        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, num_values * sizeof(float)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_values * sizeof(float)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_velocities, &tmp_pointer, sizeof(dev_velocities)));

        curandState *random_state;
        CUDA_CALL(cudaMalloc((void **)&random_state, 128 * 64 * sizeof(curandState)));
        generate_random_kernel<<<128, 64>>>(random_state);
        CUDA_CALL(cudaFree(random_state));
    }

    void suspend_simulation() {
        float *pointer;
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_velocities, sizeof(dev_velocities)));
        CUDA_CALL(cudaFree(pointer));
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_norm_coordinates, sizeof(dev_norm_coordinates)));
        CUDA_CALL(cudaFree(pointer));
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_velocities, sizeof(dev_velocities)));
        CUDA_CALL(cudaFree(pointer));
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_gravities, sizeof(dev_gravities)));
        CUDA_CALL(cudaFree(pointer));
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_distance_thresholds, sizeof(dev_distance_thresholds)));
        CUDA_CALL(cudaFree(pointer));
        CUDA_CALL(cudaMemcpyFromSymbol(&pointer, dev_forces, sizeof(dev_distance_thresholds)));
        CUDA_CALL(cudaFree(pointer));
        free(host_norm_coordinates);
    }

    void simulate_next_frame() {
        calculate_forces_kernel<512><<<4092, 512>>>();
        calculate_new_coordinates_kernel<<<512, 256>>>();
    }

    float *get_coordinates() {
        normalize_coordinates_kernel<<<64, 512>>>();

        unsigned int total_values = config->num_point_clouds * config->points_per_cloud * host_dimensions;
        float *dev_norm_coordinates_pointer;
        CUDA_CALL(cudaMemcpyFromSymbol(&dev_norm_coordinates_pointer, dev_norm_coordinates, sizeof(dev_norm_coordinates)));
        CUDA_CALL(cudaMemcpy(host_norm_coordinates, dev_norm_coordinates_pointer, total_values * sizeof(float), cudaMemcpyDeviceToHost));
        return host_norm_coordinates;
    }
}

