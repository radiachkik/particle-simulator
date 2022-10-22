#include "../include/simulation.hpp"
#include "../include/utils.h"
#include <curand_kernel.h>
#include <iostream>
#include <cuda_fp16.h>

namespace simulation {
    namespace {
        __constant__ half2 *dev_coordinates;
        __constant__ half2 *dev_velocities;
        __constant__ half2 *dev_forces;

        __constant__ half2 *dev_gravities;
        __constant__ half2 *dev_distance_thresholds;

        __constant__ float *dev_norm_coordinates;

        __constant__ unsigned int dev_num_point_clouds;
        __constant__ unsigned int dev_points_per_cloud;
        __constant__ bool dev_border_enabled;
        __constant__ half2 dev_min_distance_threshold;

        simulationConfig *config;
        float *host_norm_coordinates;

        __global__
        void generate_random_kernel(curandState *rand_state)
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;

            unsigned int total_values = dev_num_point_clouds * dev_points_per_cloud;

            curand_init(134, id, 0, &rand_state[id]);
            curandState localState = rand_state[id];

            for(unsigned int i = id; i < total_values; i += total_threads) {
                dev_coordinates[i] = __floats2half2_rn(curand_uniform(&localState) * 900 - 450, curand_uniform(&localState) * 900 - 450);
            }
        }

        __global__
        void normalize_coordinates_kernel()
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;
            unsigned int num_points = dev_points_per_cloud * dev_num_point_clouds;
            half2 scale_factor = __float2half2_rn(500.0f);
            half2 normalized_coordinate;

            for(unsigned int i = id; i < num_points; i += total_threads) {
                normalized_coordinate = __h2div(dev_coordinates[i], scale_factor);
                dev_norm_coordinates[i * 2] =  __low2float(normalized_coordinate);
                dev_norm_coordinates[i * 2 + 1] =  __high2float(normalized_coordinate);
            }
        }

        template <unsigned int block_size_x, unsigned int block_size_y>
        __global__
        void calculate_forces_kernel()
        {
            __shared__ half2 shared_force[block_size_y][block_size_x];
            __shared__ half2 partial_pc1[block_size_y];
            __shared__ half2 partial_pc2[block_size_x];

            auto thread_y_id = threadIdx.y + blockIdx.y * blockDim.y;
            auto thread_x_id = threadIdx.x + blockIdx.x * blockDim.x;

            auto num_y_threads = gridDim.y * blockDim.y;
            auto num_x_threads = gridDim.x * blockDim.x;

            unsigned int num_points = dev_points_per_cloud * dev_num_point_clouds;

            // Each y block is responsible for calculating the summed forces applied on subset pc1
            for (auto p1_index = thread_y_id; p1_index < num_points; p1_index += num_y_threads) {
                if (threadIdx.x == 0) {
                    partial_pc1[threadIdx.y] = dev_coordinates[p1_index];
                }
                __syncthreads();
                half2 force = __float2half2_rn(0.0f);
                half2 p1 = partial_pc1[threadIdx.y];
                auto pc1_index = p1_index / dev_points_per_cloud;
                // Each x block is responsible for calculating the forces between pc1 and a subset pc2
                for(auto p2_index = thread_x_id; p2_index < num_points; p2_index += num_x_threads) {
                    if (threadIdx.y == 0) {
                        partial_pc2[threadIdx.x] = dev_coordinates[p2_index];
                    }
                    __syncthreads();

                    half2 p2 = partial_pc2[threadIdx.x];
                    auto pc2_index = p2_index / dev_points_per_cloud;
                    half2 gravity = dev_gravities[pc1_index * dev_num_point_clouds + pc2_index];
                    half2 distance_threshold = dev_distance_thresholds[pc1_index * dev_num_point_clouds + pc2_index];
                    half2 distance = p2 - p1;
                    half2 r = distance * distance;
                    r = __low2half2(r) + __high2half2(r);
                    r = h2sqrt(r);
                    if (r < distance_threshold && r >= dev_min_distance_threshold) {
                        force += distance / r * gravity;
                    }
                }

                // Each y thread represents one p1. Each x thread is a subset of p2's.
                shared_force[threadIdx.y][threadIdx.x] = force;
                __syncthreads();


                // Sum up forces in shared memory

                for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        shared_force[threadIdx.y][threadIdx.x] += shared_force[threadIdx.y][threadIdx.x + s];
                    }
                    __syncthreads();
                }

                // Copy forces to global memory
                if (threadIdx.x == 0) {
                    dev_forces[p1_index + threadIdx.x] = shared_force[threadIdx.y][threadIdx.x];
                }
            }
        }

        __global__
        void calculate_new_coordinates_kernel()
        {
            unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
            unsigned int total_threads = blockDim.x * gridDim.x;
            unsigned int total_values = dev_points_per_cloud * dev_num_point_clouds;
            half2 lower_border = __float2half2_rn(-450.0f);
            half2 upper_border = __float2half2_rn(450.0f);
            half2 momentum = __float2half2_rn(0.5f);
            half2 two = __float2half2_rn(2.0f);

            for(unsigned int i = id; i < total_values; i += total_threads) {
                half2 new_velocity = (dev_velocities[i] + dev_forces[i]) * momentum;
                half2 new_coordinate = dev_coordinates[i] + new_velocity;
                if (dev_border_enabled) {
                    half2 is_below_border = __hle2(new_coordinate, lower_border);
                    half2 is_above_border = __hge2(new_coordinate, upper_border);
                    half2 is_outside_border = is_below_border + is_above_border;
                    new_velocity -= new_velocity * is_outside_border * two;
                    new_coordinate -= new_coordinate * is_outside_border;
                    new_coordinate += is_below_border * lower_border + is_above_border * upper_border;
                }
                dev_velocities[i] = new_velocity;
                dev_coordinates[i] = new_coordinate;
            }
        }
    }

    void initialize_simulation(simulationConfig *simulation_config) {
        config = simulation_config;
        const unsigned int num_points = config->num_point_clouds * config->points_per_cloud;
        const unsigned int num_values = num_points * 2;
        const unsigned int num_point_cloud_combinations = config->num_point_clouds * config->num_point_clouds;

        host_norm_coordinates = (float*) malloc(num_points * 2 * sizeof(float));

        half2 distance_threshold = __float2half2_rn(1.0f);
        CUDA_CALL(cudaMemcpyToSymbol(dev_min_distance_threshold, &distance_threshold, sizeof(dev_min_distance_threshold)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_num_point_clouds, &config->num_point_clouds, sizeof(dev_num_point_clouds)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_points_per_cloud, &config->points_per_cloud, sizeof(dev_num_point_clouds)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_border_enabled, &config->border_enabled, sizeof(dev_border_enabled)));

        half2 *tmp_pointer;

        auto* converted_distance_thresholds = (half2*) malloc(num_point_cloud_combinations * sizeof(half2));
        for (int i = 0; i < num_point_cloud_combinations; i++) {
            converted_distance_thresholds[i] = __float2half2_rn(config->distance_thresholds[i]);
        }
        CUDA_CALL(cudaMalloc(&tmp_pointer, num_point_cloud_combinations * sizeof(half2)));
        CUDA_CALL(cudaMemcpy(tmp_pointer, converted_distance_thresholds, num_point_cloud_combinations * sizeof(half2), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(dev_distance_thresholds, &tmp_pointer, sizeof(dev_distance_thresholds)));

        auto* converted_gravities = (half2*) malloc(num_point_cloud_combinations * sizeof(half2));
        for (int i = 0; i < num_point_cloud_combinations; i++) {
            converted_gravities[i] = __float2half2_rn(config->gravities[i]);
        }
        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, num_point_cloud_combinations * sizeof(half2)));
        CUDA_CALL(cudaMemcpy(tmp_pointer, converted_gravities, num_point_cloud_combinations * sizeof(half2), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyToSymbol(dev_gravities, &tmp_pointer, sizeof(dev_gravities)));

        CUDA_CALL(cudaMalloc(&tmp_pointer, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_coordinates, &tmp_pointer, sizeof(dev_coordinates)));

        CUDA_CALL(cudaMalloc(&tmp_pointer, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_forces, &tmp_pointer, sizeof(dev_forces)));

        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, num_values * sizeof(float)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_values * sizeof(float)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_norm_coordinates, &tmp_pointer, sizeof(dev_norm_coordinates)));

        CUDA_CALL(cudaMalloc((void **)&tmp_pointer, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemset(tmp_pointer, 0, num_points * sizeof(half2)));
        CUDA_CALL(cudaMemcpyToSymbol(dev_velocities, &tmp_pointer, sizeof(dev_velocities)));

        curandState *random_state;
        CUDA_CALL(cudaMalloc((void **)&random_state, 128 * 64 * sizeof(curandState)));
        generate_random_kernel<<<128, 64>>>(random_state);
        CUDA_CALL(cudaFree(random_state));
    }

    void suspend_simulation() {
        half *pointer;
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
        dim3 gridDim(64, 64);
        calculate_forces_kernel<32, 32><<<gridDim, blockDim>>>();
        calculate_new_coordinates_kernel<<<512, 256>>>();
    }

    float *get_coordinates() {
        normalize_coordinates_kernel<<<64, 512>>>();

        unsigned int total_values = config->num_point_clouds * config->points_per_cloud * 2;
        float *dev_norm_coordinates_pointer;
        CUDA_CALL(cudaMemcpyFromSymbol(&dev_norm_coordinates_pointer, dev_norm_coordinates, sizeof(dev_norm_coordinates)));
        CUDA_CALL(cudaMemcpy(host_norm_coordinates, dev_norm_coordinates_pointer, total_values * sizeof(float), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        float *foo = host_norm_coordinates;
        return host_norm_coordinates;
    }
}

