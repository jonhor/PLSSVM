#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_PRECONDITIONERS_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_PRECONDITIONERS_HPP
#pragma once

#include "plssvm/constants.hpp"
#include "plssvm/matrix.hpp"
#include "plssvm/preconditioner_types.hpp"

#include "linalg.hpp"
#include <functional>

namespace plssvm::sycl::detail::precond {

/*
 * Jacobi (diagonal) Preconditioner
 */
preconditioner_func jacobi(::sycl::queue &queue, const matrix_view<matrix_type::upper> &K, matrix_view<matrix_type::upper> &M) {
    linalg::jacobi(queue, K, M);

    preconditioner_func apply_preconditioner = [=, &queue](const soa_matrix<real_type> &R, soa_matrix<real_type> &S) -> void {
        PLSSVM_ASSERT(R.padding().x == R.padding().y, "padding is expected to be symmetric");
        PLSSVM_ASSERT(S.padding().x == S.padding().y, "padding is expected to be symmetric");

        auto R_ = utility::create_managed_view<matrix_type::general>(queue, R.data(), R.shape().y, R.shape().x, R.padding().x);
        auto S_ = utility::create_managed_view<matrix_type::general>(queue, S.data(), S.shape().y, S.shape().x, S.padding().x);

        linalg::blas::symm(queue, M, R_.view(), S_.view());

        // Copy the result back
        queue.memcpy(S.data(), S_->data(), S_->size_bytes_padded()).wait();
    };

    return apply_preconditioner;
}

/*
 * Cholesky Preconditioner
 *
 * Calculates the upper cholesky decomposition U, so that
 * K = U^T U
 *
 * And applies it by using forward / back substitution to solve S = M * R
 * Y = triangular_solve(U.T, R)
 * S = triangular_solve(U, Y)
 */
inline preconditioner_func cholesky(::sycl::queue &queue, const matrix_view<matrix_type::upper> &K, matrix_view<matrix_type::upper> &M) {
    PLSSVM_ASSERT(utility::is_valid(K), "kernel matrix K contains nan or inf values");
    PLSSVM_ASSERT(utility::is_valid(M), "precondition matrix M contains nan or inf values");

    // utility::dump_view_to_file(K, "kernel.bytes");
    linalg::chol(queue, K, M);
    // utility::dump_view_to_file(M, "cholesky.bytes");

    fmt::println("cholesky done");
    PLSSVM_ASSERT(utility::is_valid(M), "precondition matrix M contains nan or inf values");
    fmt::println("valid check is done");

    preconditioner_func apply_preconditioner = [=, &queue](const soa_matrix<real_type> &R, soa_matrix<real_type> &S) -> void {
        PLSSVM_ASSERT(R.padding().x == R.padding().y, "padding is expected to be symmetric");
        PLSSVM_ASSERT(S.padding().x == S.padding().y, "padding is expected to be symmetric");
        
        // R and S are stored in host memory, so
        // copy data to device and return a matrix view
        // the shapes of R and S are transposed (for performance reasons?)
        // but the elements are stored in row-major order in memory so we can just swap x and y here
        auto R_ = utility::create_managed_view<matrix_type::general>(queue, R.data(), R.shape().y, R.shape().x, R.padding().x);
        auto S_ = utility::create_managed_view<matrix_type::general>(queue, S.data(), S.shape().y, S.shape().x, S.padding().x);

        // Y = solve_triangular(M.T, R)
        auto Y = utility::zeros<matrix_type::general>(queue, R_->shape, R_->padding);
        auto MT = utility::transpose(queue, M);
        linalg::blas::triangular_solve_lower(queue, MT.view(), R_.view(), Y.view());

        // S = solve_triangular(M, Y)
        linalg::blas::triangular_solve_upper(queue, M, Y.view(), S_.view());

        // Copy the result back
        queue.memcpy(S.data(), S_->data(), S_->size_bytes_padded()).wait();
    };

    return apply_preconditioner;
}

}  // namespace plssvm::sycl::detail::precond

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_PRECONDITIONERS_HPP
