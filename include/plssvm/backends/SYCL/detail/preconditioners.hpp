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
 * Cholesky Preconditioner
 *
 * Calculates the upper cholesky decomposition U, so that
 * K = U^T U
 *
 * And applies it by using forward / back substitution to solve S = M * R
 * Y = triangular_solve(U.T, R)
 * S = triangular_solve(U, Y)
 */
preconditioner_func cholesky(const matrix_view<matrix_type::upper> &K, matrix_view<matrix_type::upper> &M, ::sycl::queue &queue) {
    linalg::direct_cholesky(K, M, queue);

    preconditioner_func apply_preconditioner = [=, &queue](const soa_matrix<real_type> &R, soa_matrix<real_type> &S) -> void {
        PLSSVM_ASSERT(R.padding().x == R.padding().y, "padding is expected to be symmetric");
        PLSSVM_ASSERT(S.padding().x == S.padding().y, "padding is expected to be symmetric");

        // copy data to device and return a matrix view
        auto R_view = helper::create_shared_view(R.data(), R.shape().x, R.shape().y, R.padding().x, queue);
        auto S_view = helper::create_shared_view(S.data(), S.shape().x, S.shape().y, S.padding().x, queue);

        // Y = solve_triangular(M.T, R)
        auto Y = helper::zeros_like(R_view, queue);
        auto MT = helper::transpose(M, queue);
        linalg::blas::trsm(MT, R_view, Y, queue);

        // S = solve_triangular(M, Y)
        linalg::blas::trsm(M, Y, S_view, queue);

        // Copy the result back
        queue.memcpy(S.data(), S_view.data(), S_view.size_bytes()).wait();

        ::sycl::free(R_view.data(), queue);
        ::sycl::free(S_view.data(), queue);
        ::sycl::free(Y.data(), queue);
        ::sycl::free(MT.data(), queue);
    };

    return apply_preconditioner;
}

}  // namespace plssvm::sycl::detail::precond

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_PRECONDITIONERS_HPP
