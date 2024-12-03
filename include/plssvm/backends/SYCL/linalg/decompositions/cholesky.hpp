#ifndef PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_

#include "plssvm/backends/SYCL/detail/block.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"
#include "plssvm/matrix.hpp"

#include "sycl/sycl.hpp"  // ::sycl::range, ::sycl::nd_range, ::sycl::handler, ::sycl::info::device

#include <algorithm>
#include <chrono>
#include <execution>
#include <fmt/core.h>
#include <functional>
#include <numeric>
#include <optional>

namespace plssvm::sycl::linalg {

namespace internal {

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

}  // namespace internal

/*
 * Implements a blocked (tiled) cholesky decomposition such that A = U.T * U,
 * - A is a symmetric positive semi-definite matrix, stored as an upper triangular matrix
 * - U is an upper triangular matrix
 *
 * TODO add check for psd property
 */
inline void cholesky(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &A, matrix_view<matrix_type::upper> &U, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.shape == U.shape, "A und U should have the same shape");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    block_size = 8;

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
        auto factorize_event = internal::factorize_block(queue, U, block{ block_size, row, row });
        queue.wait();

        const std::chrono::steady_clock::time_point factorize_end_time = std::chrono::steady_clock::now();
        total_factorization_time += std::chrono::duration_cast<std::chrono::milliseconds>(factorize_end_time - factorize_start_time);

        // solve all blocks in the same row
        solve_events.clear();
        const std::chrono::steady_clock::time_point solve_start_time = std::chrono::steady_clock::now();
        for (std::size_t col = row + block_size; col < N; col += block_size) {
            // can also be optimized by using a row-wide approach, e.g. having to cache the factorized diagonal block only once
            auto solve_event = internal::solve_block(queue, U, block{ block_size, row, col }, factorize_event);
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

            internal::update_trailing_block_row(queue, U, block{ block_size, trailing_row, trailing_row }, blocks_in_row, row);
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

inline matrix<matrix_type::upper> cholesky(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &A, std::size_t block_size = 0) {
    auto U = linalg::empty<matrix_type::upper>(queue, A.n_rows, A.n_cols, A.padding);
    cholesky(queue, A, U, block_size);
    return U;
}

};  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_
