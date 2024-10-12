#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
#pragma once

#include "plssvm/detail/assert.hpp"
#include "plssvm/matrix.hpp"

#include "sycl/sycl.hpp"  // ::sycl::range, ::sycl::nd_range, ::sycl::handler, ::sycl::info::device

#include "matrix_view.hpp"
#include <fmt/core.h>

namespace plssvm::sycl::detail::linalg {

namespace blas {

/*
 * Symmetric rank-k update (SYRK)
 *
 * C = alpha * A * A^T = beta * C, where
 * - A is a general (n x k) matrix
 * - C is a symmetric (n x n) matrix, stored as a upper triangular matrix
 */
inline matrix_view<matrix_type::general> syrk(const matrix_view<matrix_type::general> &A, const matrix_view<matrix_type::upper> &C, ::sycl::queue &queue);

/*
 * Triangular solve with multiple right-hand sides (TRSM)
 *
 * Solves A * X = alpha * B for X, where
 * - A is either a lower or upper triangular matrix
 * - B is a general (n x k) matrix
 *
 * TODO enable_if only for triangular matrix types
 */
template <matrix_type T>
inline void trsm(const matrix_view<T> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X, ::sycl::queue &queue) {
    PLSSVM_ASSERT(B.n_rows == X.n_rows && B.n_cols == X.n_cols, "shape of B and X must be equal");

    if constexpr (T == matrix_type::lower) {  // perform a forward solve
        queue.submit([&](::sycl::handler &cgh) {
            cgh.single_task([=]() {
                for (std::size_t col = 0; col < B.n_cols; ++col) {
                    for (std::size_t row = 0; row < B.n_rows; ++row) {
                        // calculate dot product
                        real_type dot = 0;
                        for (std::size_t k = 0; k < row; ++k) {
                            dot += A(row, k) * X(k, col);
                        }

                        X(row, col) = (B(row, col) - dot) / A(row, row);
                    }
                }
            });
        });

    } else if constexpr (T == matrix_type::upper) {  // perform a backward solve
        queue.submit([&](::sycl::handler &cgh) {
            cgh.single_task([=]() {
                for (std::size_t col = 0; col < B.n_cols; ++col) {
                    for (std::size_t row = 0; row < B.n_rows; ++row) {
                        std::size_t r = (B.n_rows - 1) - row;  // start from the last row

                        // calculate dot product
                        real_type dot = 0;
                        for (std::size_t k = r + 1; k < B.n_rows; ++k) {
                            dot += A(r, k) * X(k, col);
                        }

                        X(r, col) = (B(r, col) - dot) / A(r, r);
                    }
                }
            });
        });
    }
    queue.wait();
}

/*
 * Symmetric matrix multiplication (SYMM)
 *
 * C = alpha * A * B + beta * C
 *
 * TODO call run_blas_level_3 internally
 */
inline matrix_view<matrix_type::upper> symm(real_type alpha, matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &B, real_type beta, matrix_view<matrix_type::upper> &C);

}  // namespace blas

constexpr std::size_t BLOCK_SIZE = 1024;

inline void print_device_matrix(const matrix_view<matrix_type::upper> &U, ::sycl::queue &queue) {
    auto num_elements = U.size();

    std::vector<real_type> U_host(num_elements);
    queue.memcpy(U_host.data(), U.data(), U.size_bytes()).wait();

    for (auto elem : U_host) {
        fmt::print("{} ", elem);
    }
    fmt::println("");
}

/*
 * This directly computes the cholesky factorization element-wise
 */
inline void direct_cholesky(const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &U, ::sycl::queue &queue) {
    auto n = A.n_rows;

    queue.submit([&](::sycl::handler &cgh) {
        cgh.single_task([=]() {
            // TODO optimization?
            // first compute the diagonal element
            // then update all other elements in the same row in parallel

            for (std::size_t row = 0; row < n; ++row) {
                for (std::size_t col = row; col < n; ++col) {
                    // compute dot product
                    real_type sum = 0;
                    for (std::size_t k = 0; k < row; ++k) {
                        sum += U(k, row) * U(k, col);
                    }

                    if (row == col) {
                        U(row, col) = std::sqrt(A(row, row) - sum);
                    } else {
                        U(row, col) = (A(row, col) - sum) / U(row, row);
                    }
                }
            }
        });
    });
    queue.wait();
}

/*
 * Computes the Cholesky factorization A = U^T * U, where
 * - A is a symmetric positive definitive matrix, stored as an upper triangular matrix.
 * - U is an upper triangular matrix
 *
 * see dpotrf from lapack
 */
// inline void cholesky(matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::upper> &M, ::sycl::queue &queue) {
//     auto n = A.n_rows;
//     fmt::println("n: {}", n);
//     fmt::println("padding size: {}", PADDING_SIZE);
//     //
//     //    // TODO use the direct version first and do some testing
//     //    // with this before advancing to the block variant
//     //    fmt::println("========================");
//     //    for (std::size_t i = 0; i < A.n_rows; ++i) {
//     //        for (std::size_t j = i; j < A.n_rows; ++j) {
//     //            fmt::print("{} ", A(i, j));
//     //        }
//     //        fmt::println("");
//     //    }
//     //
//     //    auto A_byte_size = A.size_bytes();
//     //    auto U_data = ::sycl::malloc_shared<real_type>(A_byte_size, queue);
//     //    queue.memset(U_data, 0, A_byte_size).wait();
//     //    matrix_view<matrix_type::upper> U(U_data, n);
//     //
//     //    direct_cholesky(A, U, queue);
//     //
//     //    fmt::println("========================");
//     //    for (std::size_t i = 0; i < U.n_rows; ++i) {
//     //        for (std::size_t j = i; j < U.n_rows; ++j) {
//     //            fmt::print("{} ", U(i, j));
//     //        }
//     //        fmt::println("");
//     //    }
//     //    ::sycl::free(U.data(), queue);
//     //
//     //    // use the unblocked version of cholesky if order(A) <= BLOCK_SIZE
//     //    if (n <= BLOCK_SIZE) {
//     //        // direct_cholesky()
//     //    }
//
//     // perform the blocked (tiled) cholesky algorithm
// }

}  // namespace plssvm::sycl::detail::linalg
#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_BLAS_HPP
