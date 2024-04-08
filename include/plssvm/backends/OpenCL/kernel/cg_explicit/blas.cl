/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the OpenCL backend.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is only responsible for the rows this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__kernel void device_kernel_symm(const ulong num_rows, const ulong num_rhs, const ulong device_specific_num_rows, const ulong row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # rhs
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # rows
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong dim = 0; dim < (num_rows - row_offset); dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            // determine on which side of the diagonal we are located
            if (dim + get_local_id(1) < global_j) {
                A_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[(dim + get_local_id(1)) * (num_rows - row_offset + PADDING_SIZE) + global_j - (dim + get_local_id(1)) * (dim + get_local_id(1) + 1) / 2];
            } else {
                A_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[global_j * (num_rows - row_offset + PADDING_SIZE) + dim + get_local_id(1) - global_j * (global_j + 1) / 2];
            }
            // determine on which side of the diagonal we are located
            if (dim + get_local_id(1) + THREAD_BLOCK_SIZE < global_j) {
                A_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows - row_offset + PADDING_SIZE) + global_j - (dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (dim + get_local_id(1) + THREAD_BLOCK_SIZE + 1) / 2];
            } else {
                A_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[global_j * (num_rows - row_offset + PADDING_SIZE) + dim + get_local_id(1) + THREAD_BLOCK_SIZE - global_j * (global_j + 1) / 2];
            }

            B_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = B[(dim + row_offset + get_local_id(1)) * (num_rhs + PADDING_SIZE) + global_i];
            B_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = B[(dim + row_offset + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rhs + PADDING_SIZE) + global_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + internal_i;
            const ulong device_global_j = j + internal_j;
            const ulong global_j = row_offset + j + internal_j;

            if (global_i < num_rhs && device_global_j < device_specific_num_rows) {
                C[global_j * (num_rhs + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE) + global_i];
            }
        }
    }
}

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @details In a multi-GPU setting, this function is responsible for mirroring down the columns this device is responsible for!
 * @param[in] num_rows the number of rows in @p A and @p C
 * @param[in] num_rhs the number of columns in @p B and @p C
 * @param[in] num_mirror_rows the number of rows to mirror down
 * @param[in] device_specific_num_rows the number of rows in @p A and number of rows in @p B; thr rows in @p A are potentially distributed across multiple devices
 * @param[in] row_offset the first row this device is responsible for
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
__kernel void device_kernel_symm_mirror(const ulong num_rows, const ulong num_rhs, const ulong num_mirror_rows, const ulong device_specific_num_rows, const ulong row_offset, const real_type alpha, const __global real_type *A, const __global real_type *B, const real_type beta, __global real_type *C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # rhs
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # rows
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type A_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type B_cache[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

    for (ulong dim = 0; dim < device_specific_num_rows; dim += FEATURE_BLOCK_SIZE) {
        // load data into shared memory
        for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
            const ulong global_i = i_linear + internal * THREAD_BLOCK_SIZE;
            const ulong global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

            // determine on which side of the diagonal we are located
            A_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[(dim + get_local_id(1)) * (num_rows - row_offset + PADDING_SIZE) - (dim + get_local_id(1) - 1) * (dim + get_local_id(1)) / 2 + device_specific_num_rows - (dim + get_local_id(1)) + global_j];
            A_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = A[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows - row_offset + PADDING_SIZE) - (dim + get_local_id(1) + THREAD_BLOCK_SIZE - 1) * (dim + get_local_id(1) + THREAD_BLOCK_SIZE) / 2 + device_specific_num_rows - (dim + get_local_id(1) + THREAD_BLOCK_SIZE) + global_j];

            B_cache[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = B[(dim + row_offset + get_local_id(1)) * (num_rhs + PADDING_SIZE) + global_i];
            B_cache[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = B[(dim + row_offset + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rhs + PADDING_SIZE) + global_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // calculation
        for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
            for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    temp[internal_i][internal_j] += A_cache[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j] * B_cache[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + internal_i;
            const ulong partial_global_j = j + internal_j;
            const ulong global_j = row_offset + device_specific_num_rows + j + internal_j;

            if (global_i < num_rhs && partial_global_j < num_mirror_rows) {
                C[global_j * (num_rhs + PADDING_SIZE) + global_i] = alpha * temp[internal_i][internal_j] + beta * C[global_j * (num_rhs + PADDING_SIZE) + global_i];
            }
        }
    }
}

/**
 * @brief Perform a simple inplace matrix addition: lhs += rhs.
 * @param[in] num_cols the number of columns in both matrices
 * @param[in,out] lhs the first matrix (updated inplace)
 * @param[in] rhs the second matrix
 */
__kernel void device_kernel_inplace_matrix_add(const ulong num_cols, real_type __global *lhs, const real_type __global *rhs) {
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # num_rows
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + internal_i;
            const ulong global_j = j + internal_j;

            lhs[global_i * (num_cols + PADDING_SIZE) + global_j] += rhs[global_i * (num_cols + PADDING_SIZE) + global_j];
        }
    }
}

/**
 * @brief Perform a simple inplace matrix scale: lhs *= scalar.
 * @param[in] num_cols the number of columns in the matrix
 * @param[in,out] lhs the matrix (updated inplace)
 * @param[in] scale the value to scale
 */
__kernel void device_kernel_inplace_matrix_scale(const ulong num_cols, real_type __global *lhs, const real_type scale) {
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;  // # num_rows
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;  // # num_rhs

    for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
        for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
            const ulong global_i = i + internal_i;
            const ulong global_j = j + internal_j;

            lhs[global_i * (num_cols + PADDING_SIZE) + global_j] *= scale;
        }
    }
}