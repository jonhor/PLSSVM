/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the HIP backend.
 */

#ifndef PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
#define PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::FEATURE_BLOCK_SIZE

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

namespace plssvm::hip {

/**
 * @brief Create the explicit kernel matrix using the linear kernel function (\f$\vec{u}^T \cdot \vec{v}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 */
__global__ void device_kernel_assembly_linear(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j_cached_idx = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE) {
                data_cache_i[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                data_cache_j[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            }

            // load data into shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE && dim + threadIdx.y < num_features) {
                if (i < num_rows) {
                    data_cache_i[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + j_cached_idx];
                }
            }
            __syncthreads();

            // calculation
            for (unsigned long long block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += data_cache_i[block_dim][threadIdx.x] * data_cache_j[block_dim][threadIdx.y];
            }
            __syncthreads();
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = temp + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

#if defined(PLSSVM_USE_GEMM)
            ret[j * num_rows + i] = temp;
            ret[i * num_rows + j] = temp;
#else
            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
        }
    }
}

/**
 * @brief Create the explicit kernel matrix using the polynomial kernel function (\f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] degree parameter used in the polynomial kernel function
 * @param[in] gamma parameter used in the polynomial kernel function
 * @param[in] coef0 parameter used in the polynomial kernel function
 */
__global__ void device_kernel_assembly_polynomial(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j_cached_idx = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE) {
                data_cache_i[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                data_cache_j[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            }

            // load data into shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE && dim + threadIdx.y < num_features) {
                if (i < num_rows) {
                    data_cache_i[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + j_cached_idx];
                }
            }
            __syncthreads();

            // calculation
            for (unsigned long long block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += data_cache_i[block_dim][threadIdx.x] * data_cache_j[block_dim][threadIdx.y];
            }
            __syncthreads();
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = pow(gamma * temp + coef0, (double) degree) + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

#if defined(PLSSVM_USE_GEMM)
            ret[j * num_rows + i] = temp;
            ret[i * num_rows + j] = temp;
#else
            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
        }
    }
}

/**
 * @brief Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] gamma parameter used in the rbf kernel function
 */
__global__ void device_kernel_assembly_rbf(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const real_type gamma) {
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long j = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned long long j_cached_idx = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __shared__ real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (blockIdx.x >= blockIdx.y) {
        real_type temp{ 0.0 };
        for (unsigned long long dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE) {
                data_cache_i[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
                data_cache_j[threadIdx.y][threadIdx.x] = real_type{ 0.0 };
            }

            // load data into shared memory
            if (threadIdx.y < FEATURE_BLOCK_SIZE && dim + threadIdx.y < num_features) {
                if (i < num_rows) {
                    data_cache_i[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[threadIdx.y][threadIdx.x] = data_d[(dim + threadIdx.y) * (num_rows + 1) + j_cached_idx];
                }
            }
            __syncthreads();

            // calculation
            for (unsigned long long block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                const real_type d = data_cache_i[block_dim][threadIdx.x] - data_cache_j[block_dim][threadIdx.y];
                temp += d * d;
            }
            __syncthreads();
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = exp(-gamma * temp) + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

#if defined(PLSSVM_USE_GEMM)
            ret[j * num_rows + i] = temp;
            ret[i * num_rows + j] = temp;
#else
            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
        }
    }
}

}

#endif  // PLSSVM_BACKENDS_HIP_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIP_HPP_
