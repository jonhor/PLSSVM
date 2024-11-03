#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
#pragma once

#include "plssvm/detail/assert.hpp"
#include "plssvm/matrix.hpp"

#include "sycl/sycl.hpp"  // ::sycl::range, ::sycl::nd_range, ::sycl::handler, ::sycl::info::device

#include "block.hpp"
#include "matrix_view.hpp"
#include <algorithm>
#include <fmt/core.h>

namespace plssvm::sycl::detail::linalg {

constexpr unsigned MAX_BLOCK_SIZE = 32;
constexpr unsigned BLOCK_SIZE = std::min(PADDING_SIZE, MAX_BLOCK_SIZE);

namespace blas {

namespace internal {

/*
 * performs a matrix multiplication on the block level
 */
inline void matrix_multiplication_block(const ::sycl::local_accessor<real_type, 2> &A, const ::sycl::local_accessor<real_type, 2> &B, const ::sycl::local_accessor<real_type, 2> &C, std::size_t row, std::size_t col, std::size_t block_size) {
    real_type sum = 0;
    for (std::size_t k = 0; k < block_size; ++k) {
        sum += A[row][k] * B[k][col];
    }
    C[row][col] = sum;
}

/*
 * performs a forward solve on the block level
 */
inline void forward_solve_block(const ::sycl::local_accessor<real_type, 2> &A, const ::sycl::local_accessor<real_type, 2> &B, const ::sycl::nd_item<2> &item, std::size_t row, std::size_t col, std::size_t block_size) {
    for (std::size_t current_row = 0; current_row < block_size; ++current_row) {
        if (row == current_row) {
            real_type sum = 0;
            for (std::size_t k = 0; k < current_row; ++k) {
                sum += A[current_row][k] * B[k][col];
            }
            B[current_row][col] = (B[current_row][col] - sum) / A[current_row][current_row];
        }

        item.barrier(::sycl::access::fence_space::local_space);
    }
}

/*
 * performs a forward solve on the block level
 */
inline void backward_solve_block(const ::sycl::local_accessor<real_type, 2> &A, const ::sycl::local_accessor<real_type, 2> &B, const ::sycl::nd_item<2> &item, std::size_t row, std::size_t col, std::size_t block_size) {
    for (std::size_t j = 0; j < block_size; ++j) {
        const auto current_row = block_size - j - 1;

        if (row == current_row) {
            real_type sum = 0;
            for (std::size_t k = current_row + 1; k < block_size; ++k) {
                sum += A[current_row][k] * B[k][col];
            }
            B[current_row][col] = (B[current_row][col] - sum) / A[current_row][current_row];
        }

        item.barrier(::sycl::access::fence_space::local_space);
    }
}

}  // namespace internal

inline void fill(::sycl::queue &queue, matrix_view<matrix_type::general> &A) {
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE, "A should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = A.n_cols;

    const auto BS = 4;

    queue.submit([&](::sycl::handler &cgh) {
        const ::sycl::range<2> global_range(N, M);
        const ::sycl::range<2> local_range(BS, BS);
        const ::sycl::nd_range<2> nd_range(global_range, local_range);

        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            auto group_id = item.get_group().get_group_linear_id();
            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);

            A(row, col) = group_id;  // static_cast<real_type>(group_id);
        });
    });
    queue.wait();
}

/*
 * Symmetric rank-k update (SYRK)
 *
 * C = alpha * A * A^T = beta * C, where
 * - A is a general (n x k) matrix
 * - C is a symmetric (n x n) matrix, stored as a upper triangular matrix
 */
inline matrix_view<matrix_type::general> syrk(const matrix_view<matrix_type::general> &A, const matrix_view<matrix_type::upper> &C, ::sycl::queue &queue);

void triangular_solve_lower(::sycl::queue &queue, const matrix_view<matrix_type::lower> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto n = A.n_rows;
    const auto m = B.n_cols;
    const double epsilon = 1e-6;

    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            real_type sum = 0;
            for (std::size_t k = 0; k < row; ++k) {
                sum += A(row, k) * X(k, col);
            }

            double scaled_epsilon = std::max(epsilon, std::abs(A(row, row)) * epsilon);  // Adjust epsilon based on the diagonal
            X(row, col) = (X(row, col) - sum) / (A(row, row) + scaled_epsilon);
        }
    }
}

void triangular_solve_upper(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto n = A.n_rows;
    const auto m = B.n_cols;
    const real_type epsilon = 1e-8;

    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto current_idx = n - i - 1;

            real_type sum = 0;
            for (std::size_t j = 0; j < i; ++j) {
                const auto current_j = n - j - 1;
                sum += A(current_idx, current_j) * X(current_j, col);
            }

            double scaled_epsilon = std::max(epsilon, std::abs(A(current_idx, current_idx)) * epsilon);  // Adjust epsilon based on the diagonal
            X(current_idx, col) = (X(current_idx, col) - sum) / (A(current_idx, current_idx) + scaled_epsilon);
        }
    }
}

void trsm(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X, std::size_t block_size = 0) {
    block_size = block_size == 0 ? BLOCK_SIZE : block_size;

    std::size_t n_rows = A.n_rows;  // n
    std::size_t n_cols = B.n_cols;  // m

    PLSSVM_ASSERT(n_rows > 0, "should have at least one row");

    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    // total blocks in X
    const auto total_blocks = n_rows / block_size + (n_rows % block_size > 0 ? 1 : 0);
    fmt::println("total blocks: {}", total_blocks);

    // iterate over all blocks in X
    for (std::size_t block = 0; block < total_blocks; ++block) {
        // backward solve starts from the last block
        const auto current_block_index = total_blocks - block - 1;
        const auto current_row_offset = current_block_index * block_size;

        // for the first (last) block we have to check how many rows there are
        auto rows_to_solve = block_size;
        if (block == 0) {
            const auto rows_in_last_block = n_rows % block_size;
            rows_to_solve = rows_in_last_block == 0 ? block_size : rows_in_last_block;
        }

        // solve the current block for X
        auto solve_block = queue.submit([&](::sycl::handler &cgh) {
            ::sycl::range<2> solve_range(rows_to_solve, block_size);
            ::sycl::nd_range execution_range(solve_range, solve_range);

            ::sycl::local_accessor<real_type, 2> a_block(solve_range, cgh);
            ::sycl::local_accessor<real_type, 2> x_block(solve_range, cgh);

            cgh.parallel_for(execution_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                const auto global_row = row + current_row_offset;
                const auto global_col = col + current_row_offset;

                // const auto global_row = item.get_local_id
                a_block[row][col] = A(global_row, global_col);
                x_block[row][col] = X(global_row, col);
                item.barrier(::sycl::access::fence_space::local_space);

                for (std::size_t k = 0; k < rows_to_solve; ++k) {
                    const auto current_row = rows_to_solve - k - 1;

                    if (row == current_row) {
                        x_block[current_row][col] /= a_block[current_row][current_row];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);

                    if (row < current_row) {
                        x_block[row][col] -= a_block[row][current_row] * x_block[current_row][col];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                X(global_row, col) = x_block[row][col];
            });
        });
        solve_block.wait();

        // we solved the uppermost block and don't need to update anything
        if (current_block_index == 0) {
            break;
        }

        auto update_blocks = queue.submit([&](::sycl::handler &cgh) {
            ::sycl::range<2> local_range(block_size, block_size);
            ::sycl::range<2> global_range(current_block_index * block_size, block_size);
            ::sycl::nd_range<2> execution_range(global_range, local_range);

            ::sycl::local_accessor<real_type, 2> a_cache(local_range, cgh);  // diagonal block from A that is multiplied with this block
            ::sycl::local_accessor<real_type, 2> x_cache(local_range, cgh);  // block from X that was last solved

            cgh.parallel_for(execution_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                auto global_row = item.get_global_id(0);

                a_cache[row][col] = A(global_row, current_row_offset + col);
                x_cache[row][col] = X(current_row_offset + row, col);

                item.barrier(::sycl::access::fence_space::local_space);

                real_type sum = 0;
                for (std::size_t k = 0; k < block_size; ++k) {
                    sum += a_cache[row][k] * x_cache[k][col];
                }

                X(global_row, col) -= sum;
            });
        });
        update_blocks.wait();
    }
}

/*
inline void trsm(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.n_rows == B.n_rows, "A and B must have the same number of rows");
    PLSSVM_ASSERT(B.n_rows == X.n_rows && B.n_cols == X.n_cols, "shape of B and X must be equal");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    PLSSVM_ASSERT(B.n_cols <= block_size, "B with more columns than block size is currently not supported, {} > {}", B.n_cols, block_size);
    PLSSVM_ASSERT(A.padding >= block_size && B.padding >= block_size && X.padding >= block_size, "padding should be at least block size");

    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    const auto total_blocks_x = (N / block_size) + (N % block_size == 0 ? 0 : 1);
    ::sycl::range<2> block_range(block_size, block_size);

    fmt::println("solving upper with bs: {}", block_size);

    std::size_t block = 0;
    for (std::size_t i = 0; i < N; i += block_size) {
        const auto remaining_blocks_x = total_blocks_x - 1 - block++;
        const auto offset = (total_blocks_x * block_size) - i - block_size;

        std::size_t rows_to_solve = block_size;
        if (i == 0) {
            auto rows_in_last_block = N % block_size;
            rows_to_solve = rows_in_last_block == 0 ? block_size : rows_in_last_block;
        }
        // fmt::println("rows to solve {}", rows_to_solve);

        // solve the current block of X
        auto solve_diagonal_block = queue.submit([&](::sycl::handler &cgh) {
            const auto solve_range = ::sycl::range<2>(rows_to_solve, block_size);

            const ::sycl::local_accessor<real_type, 2> a_cache(solve_range, cgh);
            const ::sycl::local_accessor<real_type, 2> x_cache(solve_range, cgh);

            ::sycl::nd_range<2>
                nd_range(solve_range, solve_range);

            cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                const auto global_row = offset + row;
                const auto global_col = offset + col;

                a_cache[row][col] = A(global_row, global_col);
                x_cache[row][col] = X(global_row, col);
                item.barrier(::sycl::access::fence_space::local_space);

                for (std::size_t j = 0; j < rows_to_solve; ++j) {
                    const auto current_row = rows_to_solve - j - 1;

                    // solve current row
                    x_cache[current_row][col] /= a_cache[current_row][current_row];

                    // update the remaining stuff
                    if (row < current_row) {
                        x_cache[row][col] -= a_cache[row][current_row] * x_cache[current_row][col];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                X(global_row, col) = x_cache[row][col];
            });
        });

        if (remaining_blocks_x <= 0) {
            solve_diagonal_block.wait();
            break;
        }

        // update remaining blocks in X
        auto update_remaining_blocks = queue.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(solve_diagonal_block);

            const ::sycl::local_accessor<real_type, 2> xs_cache(block_range, cgh);  // last solved block of X
            const ::sycl::local_accessor<real_type, 2> a_cache(block_range, cgh);   // block from A that is needed for the update
            ::sycl::nd_range<2>
                nd_range(::sycl::range<2>(remaining_blocks_x * block_size, block_size), block_range);

            cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                const auto global_row = item.get_global_id(0);

                a_cache[row][col] = A(global_row, col + offset);
                xs_cache[row][col] = X(row + offset, col);
                item.barrier(::sycl::access::fence_space::local_space);

                real_type sum = 0;
                for (std::size_t k = 0; k < block_size; ++k) {
                    // for (std::size_t k = 0; k < rows_to_solve; ++k) {
                    sum += a_cache[row][k] * xs_cache[k][col];
                }

                X(global_row, col) -= sum;
            });
        });
        // wait for blocks to be updated
        update_remaining_blocks.wait();
    }
}
 */

/*
template <matrix_type T>
inline void trsm(::sycl::queue &queue, const matrix_view<T> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.n_rows == B.n_rows, "A and B must have the same number of rows");
    PLSSVM_ASSERT(B.n_rows == X.n_rows && B.n_cols == X.n_cols, "shape of B and X must be equal");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    PLSSVM_ASSERT(B.n_cols <= block_size, "B with more columns than block size is currently not supported, {} > {}", B.n_cols, block_size);
    PLSSVM_ASSERT(A.padding >= block_size && B.padding >= block_size && X.padding >= block_size, "padding should be at least block size");

    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    const auto total_blocks_x = (N / block_size) + (N % block_size == 0 ? 0 : 1);
    ::sycl::range<2> block_range(block_size, block_size);
    if constexpr (T == matrix_type::lower) {  // perform a forward solve

        std::size_t block = 0;
        for (std::size_t offset = 0; offset < N; offset += block_size) {
            const auto remaining_blocks_x = total_blocks_x - 1 - block++;

            // solve the current block of X
            auto solve_diagonal_block = queue.submit([&](::sycl::handler &cgh) {
                const ::sycl::local_accessor<real_type, 2> a_cache(block_range, cgh);
                const ::sycl::local_accessor<real_type, 2> x_cache(block_range, cgh);
                ::sycl::nd_range<2> nd_range(block_range, block_range);

                cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                    const auto row = item.get_local_id(0);
                    const auto col = item.get_local_id(1);
                    const auto global_row = row + offset;
                    const auto global_col = col + offset;

                    a_cache[row][col] = A(global_row, global_col);
                    x_cache[row][col] = X(global_row, col);
                    item.barrier(::sycl::access::fence_space::local_space);

                    internal::forward_solve_block(a_cache, x_cache, item, row, col, block_size);

                    X(global_row, col) = x_cache[row][col];
                });
            });

            if (remaining_blocks_x <= 0) {
                solve_diagonal_block.wait();
                break;
            }

            // update remaining blocks in X
            queue.submit([&](::sycl::handler &cgh) {
                cgh.depends_on(solve_diagonal_block);

                const ::sycl::local_accessor<real_type, 2> xs_cache(block_range, cgh);  // last solved block of X
                const ::sycl::local_accessor<real_type, 2> a_cache(block_range, cgh);   // block from A that is needed for the update
                ::sycl::nd_range<2>
                    nd_range(::sycl::range<2>(remaining_blocks_x * block_size, block_size), block_range);

                cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                    const auto row = item.get_local_id(0);
                    const auto col = item.get_local_id(1);
                    const auto global_row = item.get_global_id(0) + offset + block_size;

                    a_cache[row][col] = A(global_row, col + offset);
                    xs_cache[row][col] = X(row + offset, col);
                    item.barrier(::sycl::access::fence_space::local_space);

                    real_type sum = 0;
                    for (std::size_t k = 0; k < block_size; ++k) {
                        sum += a_cache[row][k] * xs_cache[k][col];
                    }

                    X(global_row, col) -= sum;
                });
            });
            // wait for blocks to be updated
            queue.wait();
        }

    } else if constexpr (T == matrix_type::upper) {  // perform a backward solve

        fmt::println("solving upper with bs: {}", block_size);

        std::size_t block = 0;
        for (std::size_t i = 0; i < N; i += block_size) {
            const auto remaining_blocks_x = total_blocks_x - 1 - block++;
            const auto offset = (total_blocks_x * block_size) - i - block_size;

            std::size_t rows_to_solve = block_size;
            if (i == 0) {
                auto rows_in_last_block = N % block_size;
                rows_to_solve = rows_in_last_block == 0 ? block_size : rows_in_last_block;
            }
            // fmt::println("rows to solve {}", rows_to_solve);

            // solve the current block of X
            auto solve_diagonal_block = queue.submit([&](::sycl::handler &cgh) {
                const auto solve_range = ::sycl::range<2>(rows_to_solve, block_size);

                const ::sycl::local_accessor<real_type, 2> a_cache(solve_range, cgh);
                const ::sycl::local_accessor<real_type, 2> x_cache(solve_range, cgh);

                ::sycl::nd_range<2>
                    nd_range(solve_range, solve_range);

                cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                    const auto row = item.get_local_id(0);
                    const auto col = item.get_local_id(1);
                    const auto global_row = offset + row;
                    const auto global_col = offset + col;

                    a_cache[row][col] = A(global_row, global_col);
                    x_cache[row][col] = X(global_row, col);
                    item.barrier(::sycl::access::fence_space::local_space);

                    for (std::size_t j = 0; j < rows_to_solve; ++j) {
                        const auto current_row = rows_to_solve - j - 1;

                        // solve current row
                        x_cache[current_row][col] /= a_cache[current_row][current_row];

                        // update the remaining stuff
                        if (row < current_row) {
                            x_cache[row][col] -= a_cache[row][current_row] * x_cache[current_row][col];
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }

                    X(global_row, col) = x_cache[row][col];
                });
            });

            if (remaining_blocks_x <= 0) {
                solve_diagonal_block.wait();
                break;
            }

            // update remaining blocks in X
            auto update_remaining_blocks = queue.submit([&](::sycl::handler &cgh) {
                cgh.depends_on(solve_diagonal_block);

                const ::sycl::local_accessor<real_type, 2> xs_cache(block_range, cgh);  // last solved block of X
                const ::sycl::local_accessor<real_type, 2> a_cache(block_range, cgh);   // block from A that is needed for the update
                ::sycl::nd_range<2>
                    nd_range(::sycl::range<2>(remaining_blocks_x * block_size, block_size), block_range);

                cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                    const auto row = item.get_local_id(0);
                    const auto col = item.get_local_id(1);
                    const auto global_row = item.get_global_id(0);

                    a_cache[row][col] = A(global_row, col + offset);
                    xs_cache[row][col] = X(row + offset, col);
                    item.barrier(::sycl::access::fence_space::local_space);

                    real_type sum = 0;
                    for (std::size_t k = 0; k < block_size; ++k) {
                        // for (std::size_t k = 0; k < rows_to_solve; ++k) {
                        sum += a_cache[row][k] * xs_cache[k][col];
                    }

                    X(global_row, col) -= sum;
                });
            });
            // wait for blocks to be updated
            update_remaining_blocks.wait();
        }
    }
}
*/

/*
 * General matrix multiplication (GEMM)
 * C = A * B
 */
inline void gemm(::sycl::queue &queue, const matrix_view<matrix_type::general> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) {
    PLSSVM_ASSERT(A.n_cols == B.n_rows, "A should have the same number of rows as B has columns");
    PLSSVM_ASSERT(A.n_rows == C.n_rows, "C should have the same number of rows as A");
    PLSSVM_ASSERT(B.n_cols == C.n_cols, "C should have the same number of columns as B");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "A and B should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    ::sycl::range<2> global_range(N, M);                   // total number of work items
    ::sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);  // size of each work group
    ::sycl::nd_range<2> nd_range(global_range, local_range);

    queue.submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<real_type, 2> A_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        ::sycl::local_accessor<real_type, 2> B_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            auto cache_row = item.get_local_id(0);
            auto cache_col = item.get_local_id(1);

            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);

            real_type sum = 0;

            // loop over blocks
            for (std::size_t block = 0; block < N; block += BLOCK_SIZE) {
                A_cache[cache_row][cache_col] = A(row, block + cache_col);
                B_cache[cache_row][cache_col] = B(block + cache_row, col);

                // synchronize to make sure all threads have loaded their data
                item.barrier(::sycl::access::fence_space::local_space);

                // perform multiplication for the current block, i.e.
                // dot product of a row from A and a column from B

                for (std::size_t k = 0; k < BLOCK_SIZE; ++k) {
                    sum += A_cache[cache_row][k] * B_cache[k][cache_col];
                }

                // sychronize before loading the next block
                item.barrier(::sycl::access::fence_space::local_space);
            }

            C(row, col) = sum;
        });
    });
    queue.wait_and_throw();
}

/*
 * Symmetric matrix multiplication (SYMM)
 * C = A * B, where
 * - A is a symmetric matrix stored as an upper triangular matrix
 */
inline void symm(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) {
    PLSSVM_ASSERT(A.n_cols == B.n_rows, "A should have the same number of rows as B has columns");
    PLSSVM_ASSERT(A.n_rows == C.n_rows, "C should have the same number of rows as A");
    PLSSVM_ASSERT(B.n_cols == C.n_cols, "C should have the same number of columns as B");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "A and B should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    ::sycl::range<2> global_range(N, M);                   // total number of work items
    ::sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);  // size of each work group
    ::sycl::nd_range<2> nd_range(global_range, local_range);

    queue.submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<real_type, 2> A_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        ::sycl::local_accessor<real_type, 2> B_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            auto cache_row = item.get_local_id(0);
            auto cache_col = item.get_local_id(1);

            auto row = item.get_global_id(0);
            auto col = item.get_global_id(1);

            real_type sum = 0;

            // loop over blocks
            for (std::size_t block = 0; block < N; block += BLOCK_SIZE) {
                A_cache[cache_row][cache_col] = A(row, block + cache_col);
                B_cache[cache_row][cache_col] = B(block + cache_row, col);

                // synchronize to make sure all threads have loaded their data
                item.barrier(::sycl::access::fence_space::local_space);

                // perform multiplication for the current block, i.e.
                // dot product of a row from A and a column from B

                for (std::size_t k = 0; k < BLOCK_SIZE; ++k) {
                    sum += A_cache[cache_row][k] * B_cache[k][cache_col];
                }

                // sychronize before loading the next block
                item.barrier(::sycl::access::fence_space::local_space);
            }

            C(row, col) = sum;
        });
    });
    queue.wait();
}

}  // namespace blas

/*
 * Calculates jacobi row-wise.
 */
inline void jacobi(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &D) {
    queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(::sycl::range<1>(A.n_rows), [=](const ::sycl::id<1> idx) {
            D(idx, idx) = 1.0 / A(idx, idx);
        });
    });
    queue.wait();
}

namespace cholesky {
/*
 * factorizes a diagonal block in place
 */
inline ::sycl::event factorize_block(::sycl::queue &queue, matrix_view<matrix_type::upper> &U, block b) {
    const auto N = b.size;
    const ::sycl::range<2> local_range(N, N);
    const ::sycl::nd_range<2> nd_range(local_range, local_range);

    return queue.submit([&](::sycl::handler &cgh) {
        const ::sycl::local_accessor<real_type, 2> cache(local_range, cgh);

        cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);
            const auto global_row = row + b.row_offset;
            const auto global_col = col + b.col_offset;

            if (row > col) {
                return;
            }

            // load block data into local cache
            cache[row][col] = U(global_row, global_col);
            item.barrier(::sycl::access::fence_space::local_space);

            for (std::size_t current_row = 0; current_row < N; ++current_row) {
                // compute the diagonal element for the current row
                cache[current_row][current_row] = std::sqrt(cache[current_row][current_row]);
                // update the remaining row
                if (col > current_row) {
                    cache[current_row][col] = cache[current_row][col] / cache[current_row][current_row];
                }
                item.barrier(::sycl::access::fence_space::local_space);

                // update the trailing submatrix
                if (row > current_row) {
                    cache[row][col] -= cache[current_row][col] * cache[current_row][row];
                }
                item.barrier(::sycl::access::fence_space::local_space);
            }

            // write the results back to U
            U(global_row, global_col) = cache[row][col];
        });
    });
}

inline ::sycl::event solve_block(::sycl::queue &queue, matrix_view<matrix_type::upper> &U, block b, const ::sycl::event &factorize_event) {
    const auto N = b.size;
    const ::sycl::range<2> local_range(N, N);
    const ::sycl::nd_range<2> nd_range(local_range, local_range);

    return queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(factorize_event);

        ::sycl::local_accessor<real_type, 2> b_cache(local_range, cgh);
        ::sycl::local_accessor<real_type, 2> d_cache(local_range, cgh);

        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);

            const auto global_row = row + b.row_offset;
            const auto global_col = col + b.col_offset;
            const auto diag_col = col + b.row_offset;

            b_cache[row][col] = U(global_row, global_col);

            // transpose data to perform a forward substitution
            if (row <= col) {
                d_cache[col][row] = U(global_row, diag_col);
            }

            item.barrier(::sycl::access::fence_space::local_space);

            for (std::size_t current_row = 0; current_row < N; ++current_row) {
                if (row == current_row) {
                    real_type sum = 0;
                    for (std::size_t k = 0; k < current_row; ++k) {
                        sum += d_cache[current_row][k] * b_cache[k][col];
                    }
                    b_cache[current_row][col] = (b_cache[current_row][col] - sum) / d_cache[current_row][current_row];
                }

                item.barrier(::sycl::access::fence_space::local_space);
            }

            U(global_row, global_col) = b_cache[row][col];
        });
    });
}

inline ::sycl::event solve_block_upper(::sycl::queue &queue, matrix_view<matrix_type::upper> &U, block b, const ::sycl::event &diag_block_factorized) {
    const auto N = b.size;
    const ::sycl::range<2> local_range(N, N);
    const ::sycl::nd_range<2> nd_range(local_range, local_range);

    return queue.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(diag_block_factorized);

        ::sycl::local_accessor<real_type, 2> b_cache(local_range, cgh);
        ::sycl::local_accessor<real_type, 2> d_cache(local_range, cgh);

        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);

            const auto global_row = row + b.row_offset;
            const auto global_col = col + b.col_offset;
            const auto diag_col = col + b.row_offset;

            b_cache[row][col] = U(global_row, global_col);
            if (row <= col) {
                d_cache[row][col] = U(global_row, diag_col);
            }
            item.barrier(::sycl::access::fence_space::local_space);

            for (std::size_t j = 0; j < N; ++j) {
                const auto current_row = (N - 1) - j;

                if (row == current_row) {
                    real_type sum = 0;
                    for (std::size_t k = current_row + 1; k < N; ++k) {
                        sum += d_cache[current_row][k] * b_cache[k][col];
                    }
                    b_cache[current_row][col] = (b_cache[current_row][col] - sum) / d_cache[current_row][current_row];
                }

                item.barrier(::sycl::access::fence_space::local_space);
            }

            U(global_row, global_col) = b_cache[row][col];
        });
    });
}

inline ::sycl::event update_trailing_block(::sycl::queue &queue, matrix_view<matrix_type::upper> &U, block b, std::size_t last_solved_row, const std::vector<::sycl::event> &block_solved) {
    const auto N = b.size;
    const auto block_i = b.row_offset / N;
    const auto block_j = b.col_offset / N;

    // const auto n_iter = last_solved_row / b.size;

    const ::sycl::range<2> local_range(N, N);
    const ::sycl::nd_range<2> nd_range(local_range, local_range);

    return queue.submit([&](::sycl::handler &cgh) {
        // cgh.depends_on(block_solved[block_i - n_iter]);
        // cgh.depends_on(block_solved[block_j - n_iter]);

        const ::sycl::local_accessor<real_type, 2> i_cache(local_range, cgh);
        const ::sycl::local_accessor<real_type, 2> j_cache(local_range, cgh);

        cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);
            const auto global_row = row + b.row_offset;
            const auto global_col = col + b.col_offset;

            // U_li ^ T @ U_lj, where l is the last solved row
            i_cache[col][row] = U(last_solved_row + row, b.row_offset + col);
            j_cache[row][col] = U(last_solved_row + row, b.col_offset + col);
            item.barrier(::sycl::access::fence_space::local_space);

            real_type sum = 0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += i_cache[row][k] * j_cache[k][col];
            }

            U(global_row, global_col) -= sum;
        });
    });
}

inline ::sycl::event update_trailing_block_row(::sycl::queue &queue, matrix_view<matrix_type::upper> &U, block b, std::size_t n_blocks, std::size_t last_solved_row) {
    const auto N = b.size;
    const auto block_i = b.row_offset / N;
    const auto block_j = b.col_offset / N;

    // const auto n_iter = last_solved_row / b.size;

    const ::sycl::range<2> global_range(N, N * n_blocks);
    const ::sycl::range<2> local_range(N, N);
    const ::sycl::nd_range<2> nd_range(global_range, local_range);

    return queue.submit([&](::sycl::handler &cgh) {
        const ::sycl::local_accessor<real_type, 2> i_cache(local_range, cgh);
        const ::sycl::local_accessor<real_type, 2> j_cache(local_range, cgh);

        cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);
            const auto global_row = row + b.row_offset;
            const auto global_col = item.get_global_id(1) + b.col_offset;

            // U_li ^ T @ U_lj, where l is the last solved row
            i_cache[col][row] = U(last_solved_row + row, b.row_offset + col);
            j_cache[row][col] = U(last_solved_row + row, global_col);
            item.barrier(::sycl::access::fence_space::local_space);

            real_type sum = 0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += i_cache[row][k] * j_cache[k][col];
            }

            U(global_row, global_col) -= sum;
        });
    });
}

}  // namespace cholesky

/*
 * Implements a blocked (tiled) cholesky decomposition such that A = U.T * U,
 * - A is a psd matrix, stored as an upper triangular matrix
 * - U is an upper triangular matrix
 *
 * TODO add check for psd property
 */
inline void chol(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &U, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.shape == U.shape, "A und U should have the same shape");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    PLSSVM_ASSERT(A.padding >= block_size && U.padding >= block_size, "padding should be at least block size");

    // copy A to U and then run all operations on U inplace
    queue.memcpy(U.data(), A.data(), A.size_bytes_padded()).wait();

    const auto N = A.n_rows;  // order of A and U

    fmt::println("\n--- CHOLESKY ---");
    fmt::println("block size: {}", block_size);

    std::chrono::milliseconds total_factorization_time{ 0 };
    std::chrono::milliseconds total_solve_time{ 0 };
    std::chrono::milliseconds total_update_time{ 0 };

    std::vector<::sycl::event> solve_events(N);
    for (std::size_t row = 0; row < N; row += block_size) {
        // factorize the current diagonal block
        const std::chrono::steady_clock::time_point factorize_start_time = std::chrono::steady_clock::now();
        auto factorize_event = cholesky::factorize_block(queue, U, block{ block_size, row, row });

        // remove if not profiling
        queue.wait();
        const std::chrono::steady_clock::time_point factorize_end_time = std::chrono::steady_clock::now();
        total_factorization_time += std::chrono::duration_cast<std::chrono::milliseconds>(factorize_end_time - factorize_start_time);

        // solve all blocks in the same row
        solve_events.clear();
        const std::chrono::steady_clock::time_point solve_start_time = std::chrono::steady_clock::now();
        for (std::size_t col = row + block_size; col < N; col += block_size) {
            // can also be optimized by using a row-wide approach, e.g. having to cache the factorized diagonal block only once
            auto solve_event = cholesky::solve_block(queue, U, block{ block_size, row, col }, factorize_event);
            solve_events.push_back(solve_event);
        }
        queue.wait();
        const std::chrono::steady_clock::time_point solve_end_time = std::chrono::steady_clock::now();
        total_solve_time += std::chrono::duration_cast<std::chrono::milliseconds>(solve_end_time - solve_start_time);

        // update the trailing submatrix
        const std::chrono::steady_clock::time_point update_start_time = std::chrono::steady_clock::now();
        for (std::size_t trailing_row = row + block_size; trailing_row < N; trailing_row += block_size) {
            std::size_t blocks_in_row = 0;
            for (std::size_t col = trailing_row; col < N; col += block_size) {
                blocks_in_row += 1;
            }

            cholesky::update_trailing_block_row(queue, U, block{ block_size, trailing_row, trailing_row }, blocks_in_row, row);
            // for (std::size_t col = trailing_row; col < N; col += block_size) {
            //     // TODO use a row or column-wide approach reusing cached blocks
            //     cholesky::update_trailing_block(queue, U, block{ block_size, trailing_row, col }, row, solve_events);
            // }
        }

        // wait for all operations to finish before moving on to the next iteration
        queue.wait();
        const std::chrono::steady_clock::time_point update_end_time = std::chrono::steady_clock::now();
        total_update_time += std::chrono::duration_cast<std::chrono::milliseconds>(update_end_time - update_start_time);
    }

    fmt::println("total factorize time: {}", total_factorization_time);
    fmt::println("total solve time: {}", total_solve_time);
    fmt::println("total update time: {}", total_update_time);
}

inline void simple_cholesky(const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &U, ::sycl::queue &queue) {
    auto N = A.n_rows;

    ::sycl::range<2> global_range(N, N);
    ::sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);
    ::sycl::nd_range<2> nd_range(global_range, local_range);

    queue.submit([&](::sycl::handler &h) {
        // Local memory for shared data
        ::sycl::local_accessor<float, 1> cache(::sycl::range<1>(BLOCK_SIZE), h);

        h.parallel_for<class cholesky_kernel>(nd_range, [=](::sycl::nd_item<2> item) {
            size_t row = item.get_global_id(0);
            size_t col = item.get_global_id(1);

            if (row < N && col <= row) {
                real_type sum = 0;

                if (row == col) {
                    // Diagonal elements
                    A(row, col) = std::sqrt(A(row, col) - sum);
                } else {
                    // Off-diagonal elements
                    A(row, col) = (1.0f / A(row, row) * (A(row, col) - sum));
                }

                // Synchronize work-items within the work-group
                item.barrier(::sycl::access::fence_space::local_space);
            }
        });
    });
}

}  // namespace plssvm::sycl::detail::linalg
#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
