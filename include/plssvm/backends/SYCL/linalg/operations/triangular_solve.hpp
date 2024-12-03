#ifndef PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_

#include "plssvm/backends/SYCL/linalg/constants.hpp"

namespace plssvm::sycl::linalg {

inline void triangular_solve_lower(::sycl::queue &queue, const matrix_view<matrix_type::lower> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
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

inline void triangular_solve_upper(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
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

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_
