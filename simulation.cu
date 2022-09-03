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
        __constant__ const unsigned int DEV_DIMENSIONS = 2;
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

            unsigned int total_values = dev_num_point_clouds * dev_points_per_cloud * DEV_DIMENSIONS;

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
            unsigned int total_values = dev_points_per_cloud * DEV_DIMENSIONS * dev_num_point_clouds;

            for(unsigned int i = id; i < total_values; i += total_threads) {
                dev_norm_coordinates[i] = dev_coordinates[i] / 500;
            }
        }

        template <unsigned int block_size_x, unsigned int block_size_y>
        __global__
        void calculate_forces_kernel()
        {
            __shared__ float shared_force[block_size_y][block_size_x * DEV_DIMENSIONS];
            __shared__ float partial_pc1[block_size_x * DEV_DIMENSIONS];
            __shared__ float partial_pc2[block_size_y * DEV_DIMENSIONS];

            auto thread_y_id = threadIdx.y + blockIdx.y * blockDim.y;
            auto thread_x_id = threadIdx.x + blockIdx.x * blockDim.x;

            auto num_y_threads = gridDim.y * blockDim.y;
            auto num_x_threads = gridDim.x * blockDim.x;

            unsigned int num_points = dev_points_per_cloud * dev_num_point_clouds;

            // Each y block is responsible for calculating the summed forces applied on subset pc1
            for (auto p1_index = thread_y_id; p1_index < num_points; p1_index += num_y_threads) {
                if (threadIdx.x == 0) {
                    for (auto d = 0; d < DEV_DIMENSIONS; d++) {
                        partial_pc1[threadIdx.y * DEV_DIMENSIONS + d] = dev_coordinates[p1_index * DEV_DIMENSIONS + d];
                    }
                }
                __syncthreads();
                float force[DEV_DIMENSIONS] = {0.0f};
                float *p1 = partial_pc1 + threadIdx.y * DEV_DIMENSIONS;
                auto pc1_index = p1_index / dev_points_per_cloud;
                // Each x block is responsible for calculating the forces between pc1 and a subset pc2
                for(auto p2_index = thread_x_id; p2_index < dev_points_per_cloud * dev_num_point_clouds; p2_index += num_x_threads) {
                    if (threadIdx.y == 0) {
                        for (auto d = 0; d < DEV_DIMENSIONS; d++) {
                            partial_pc2[threadIdx.x * DEV_DIMENSIONS + d] = dev_coordinates[p2_index * DEV_DIMENSIONS + d];
                        }
                    }
                    __syncthreads();

                    float *p2 = partial_pc2 + threadIdx.x * DEV_DIMENSIONS;
                    auto pc2_index = p2_index / dev_points_per_cloud;
                    float gravity = dev_gravities[pc1_index * dev_num_point_clouds + pc2_index];
                    float distance_threshold = dev_distance_thresholds[pc1_index * dev_num_point_clouds + pc2_index];
                    float distance[DEV_DIMENSIONS];
                    for (int d = 0; d < DEV_DIMENSIONS; d++) {
                        distance[d] = p2[d] - p1[d];
                    }
                    float r = normf(DEV_DIMENSIONS, distance);
                    if (r < distance_threshold && r >= 1) {
                        for (int d = 0; d < DEV_DIMENSIONS; d++) {
                            force[d] += (distance[d] / (r * 100) ) * gravity;
                        }
                    }
                }

                // Each y thread represents one p1. Each x thread is a subset of p2's.
                for (int d = 0; d < DEV_DIMENSIONS; d++) {
                    shared_force[threadIdx.y][threadIdx.x * DEV_DIMENSIONS + d] = force[d];
                }
                __syncthreads();


                // Sum up forces in shared memory

                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        for (unsigned int d = 0; d < DEV_DIMENSIONS; d++) {
                            shared_force[threadIdx.y][(threadIdx.x * DEV_DIMENSIONS) + d] += shared_force[threadIdx.y][(threadIdx.x + s) * DEV_DIMENSIONS + d];
                        }
                    }
                    __syncthreads();
                }

                // Copy forces to global memory
                if (threadIdx.x < DEV_DIMENSIONS) {
                    dev_forces[p1_index * DEV_DIMENSIONS + threadIdx.x] = shared_force[threadIdx.y][threadIdx.x];
                }
                __syncthreads();
            }
        }

        __global__
        void calculate_new_coordinates_kernel()
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;
            unsigned int total_values = dev_points_per_cloud * DEV_DIMENSIONS * dev_num_point_clouds;

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
        dim3 blockDim(32, 32);
        dim3 gridDim(128, 128);
        calculate_forces_kernel<64, 64><<<gridDim, blockDim>>>();
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

