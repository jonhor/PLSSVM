#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_ARITHMETIC_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_ARITHMETIC_HPP

#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"

namespace plssvm::sycl::linalg {

/**
 * Matrix Multiplication
 * C = A @ B
 */
template <matrix_type T, typename = std::enable_if_t<T == matrix_type::general || T == matrix_type::symmetric>>
inline void matrix_multiplication(::sycl::queue &queue, const matrix_view<T> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) {
    PLSSVM_ASSERT(A.n_cols == B.n_rows, "A should have the same number of columns as B has rows");
    PLSSVM_ASSERT(A.n_rows == C.n_rows, "C should have the same number of rows as A");
    PLSSVM_ASSERT(B.n_cols == C.n_cols, "C should have the same number of columns as B");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "A and B should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = B.n_cols;
    const auto K = A.n_cols;

    auto nd_range = detail::get_uniform_2d_range(N, M, BLOCK_SIZE);
    auto event = queue.submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<real_type, 2> A_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        ::sycl::local_accessor<real_type, 2> B_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        cgh.parallel_for(nd_range, [=](::sycl::nd_item<2> item) {
            auto row = item.get_local_id(0);
            auto col = item.get_local_id(1);

            auto global_row = item.get_global_id(0);
            auto global_col = item.get_global_id(1);

            real_type sum{ 0 };

            // loop over blocks
            for (std::size_t offset = 0; offset < K; offset += BLOCK_SIZE) {
                A_cache[row][col] = A(global_row, offset + col);
                B_cache[row][col] = B(offset + row, global_col);

                // synchronize to make sure all threads have loaded their data
                item.barrier(::sycl::access::fence_space::local_space);

                // perform multiplication for the current block, i.e.
                // dot product of a row from A and a column from B
                for (std::size_t i = 0; i < BLOCK_SIZE; ++i) {
                    sum += A_cache[row][i] * B_cache[i][col];
                }

                // synchronize before loading the next block
                item.barrier(::sycl::access::fence_space::local_space);
            }

            C(global_row, global_col) = sum;
        });
    });
    event.wait();
}

/**
 * Specialized matrix multiplication if A is diagonal.
 * C = A * B
 */
inline void matrix_multiplication(::sycl::queue &queue, const matrix_view<matrix_type::diagonal> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) {
    PLSSVM_ASSERT(A.n_rows == A.n_cols, "Diagonal matrix multiplication is currently only supported if A is a square matrix");
    PLSSVM_ASSERT(A.n_cols == B.n_rows, "A should have the same number of columns as B has rows");
    PLSSVM_ASSERT(A.n_rows == C.n_rows, "C should have the same number of rows as A");
    PLSSVM_ASSERT(B.n_cols == C.n_cols, "C should have the same number of columns as B");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "A and B should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    auto nd_range = detail::get_uniform_2d_range(N, M, BLOCK_SIZE);
    auto event = queue.submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<real_type, 1> diag_cache(::sycl::range<1>(BLOCK_SIZE), cgh);
        ::sycl::local_accessor<real_type, 2> B_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        cgh.parallel_for<class diagonal_matrix_multiplication_lhs>(nd_range, [=](::sycl::nd_item<2> item) {
            const auto local_row = item.get_local_id(0);
            const auto local_col = item.get_local_id(1);

            const auto global_row = item.get_global_id(0);
            const auto global_col = item.get_global_id(1);

            if (global_row < N && global_col < M) {
                // load diagonal elements corresponding to the current row
                if (local_col == 0) {
                    diag_cache[local_row] = A(global_row, global_row);
                }
                B_cache[local_row][local_col] = B(global_row, global_col);
            }
            // synchronize to make sure all threads have loaded their data
            item.barrier(::sycl::access::fence_space::local_space);

            if (global_row < N && global_col < M) {
                real_type value{ diag_cache[local_row] * B_cache[local_row][local_col] };
                C(global_row, global_col) = value;
            }
        });
    });
    event.wait();
}

/**
 * Specialized matrix multiplication if B is diagonal.
 * C = A * B
 */
inline void matrix_multiplication(::sycl::queue &queue, const matrix_view<matrix_type::general> &A, const matrix_view<matrix_type::diagonal> &B, matrix_view<matrix_type::general> &C) {
    PLSSVM_ASSERT(B.n_rows == B.n_cols, "Diagonal matrix multiplication is currently only supported if B is a square matrix");
    PLSSVM_ASSERT(A.n_cols == B.n_rows, "A should have the same number of columns as B has rows");
    PLSSVM_ASSERT(A.n_rows == C.n_rows, "C should have the same number of rows as A");
    PLSSVM_ASSERT(B.n_cols == C.n_cols, "C should have the same number of columns as B");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "A and B should be padded with at least PADDING_SIZE to avoid out-of-bounds access");

    const auto N = A.n_rows;
    const auto M = B.n_cols;

    auto nd_range = detail::get_uniform_2d_range(N, M, BLOCK_SIZE);
    auto event = queue.submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<real_type, 1> diag_cache(::sycl::range<1>(BLOCK_SIZE), cgh);
        ::sycl::local_accessor<real_type, 2> A_cache(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        cgh.parallel_for<class diagonal_matrix_multiplication_rhs>(nd_range, [=](::sycl::nd_item<2> item) {
            const auto local_row = item.get_local_id(0);
            const auto local_col = item.get_local_id(1);

            const auto global_row = item.get_global_id(0);
            const auto global_col = item.get_global_id(1);

            if (global_row < N && global_col < M) {
                if (local_row == 0) {
                    diag_cache[local_col] = B(global_col, global_col);
                }
                A_cache[local_row][local_col] = A(global_row, global_col);
            }
            // synchronize to make sure all threads have loaded their data
            item.barrier(::sycl::access::fence_space::local_space);

            if (global_row < N && global_col < M) {
                real_type value{ A_cache[local_row][local_col] * diag_cache[local_col] };

                C(global_row, global_col) = value;
            }
        });
    });
    event.wait();
}

template <matrix_type U, matrix_type V>
inline matrix<matrix_type::general> matrix_multiplication(::sycl::queue &queue, const matrix_view<U> &A, const matrix_view<V> &B) {
    auto C = zeros<matrix_type::general>(queue, A.n_rows, B.n_cols, PADDING_SIZE);
    matrix_multiplication(queue, A, B, C.view());
    return C;
}

/*
 * *******************************
 * * Matrix Addition
 * *******************************
 */

/**
 * Adds B to A, where B is scaled by alpha.
 * A = A * alpha * B
 */
inline void matrix_addition(::sycl::queue &queue, matrix_view<matrix_type::general> &A, real_type alpha, const matrix_view<matrix_type::general> &B) {
    PLSSVM_ASSERT(A.shape == B.shape, "A and B must have the same shape");
    PLSSVM_ASSERT(A.padding >= BLOCK_SIZE && B.padding >= BLOCK_SIZE, "block size is higher than the padding of A or B");

    const auto N = A.n_rows;
    const auto M = A.n_cols;

    auto nd_range = detail::get_uniform_2d_range(N, M, BLOCK_SIZE);
    auto event = queue.parallel_for<class matrix_addition>(nd_range, [=](::sycl::nd_item<2> item) {
        const auto global_row = item.get_global_id(0);
        const auto global_col = item.get_global_id(1);

        if (global_row < N && global_col < M) {
            A(global_row, global_col) += alpha * B(global_row, global_col);
        }
    });
    event.wait();
}

}  // namespace plssvm::sycl::linalg
#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_ARITHMETIC_HPP
