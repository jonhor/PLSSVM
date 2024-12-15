#ifndef PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"

#include <Eigen/Eigen>
#include <iostream>

namespace plssvm::sycl::linalg {

using EigenMatrixT = Eigen::Matrix<real_type, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * Compute a thin SVD, such that A = U * Sigma * VT
 * We only need to compute U and Sigma for now.
 */
using svd_return_type = std::tuple<matrix<matrix_type::general>, matrix<matrix_type::diagonal>>;

inline svd_return_type svd(::sycl::queue &queue, const matrix_view<matrix_type::general> &A) {
    PLSSVM_ASSERT(A.n_rows >= A.n_cols, "A is expected to have more rows than columns");

    const auto n = A.n_rows;
    const auto m = A.n_cols;
    const auto p = A.padding;

    // Eigen uses signed indices, while matrix_view uses std::size_t.
    EigenMatrixT M(n, m);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            auto i_ = static_cast<Eigen::Index>(i);
            auto j_ = static_cast<Eigen::Index>(j);

            M(i_, j_) = A(i, j);
        }
    }

    Eigen::BDCSVD<EigenMatrixT> singular_value_decomposition{ M, Eigen::ComputeThinU };
    // Eigen::JacobiSVD<EigenMatrixT> singular_value_decomposition{ M, Eigen::ComputeThinU };

    auto S_ = singular_value_decomposition.singularValues();
    auto U_ = singular_value_decomposition.matrixU();

    PLSSVM_ASSERT(S_.size() == m, "the number of singular values should be equal to number of columns in A");
    PLSSVM_ASSERT(U_.rows() == n && U_.cols() == m, "U is expected to have the same shape as A");

    auto S = zeros<matrix_type::diagonal>(queue, m, m, p);
    for (std::size_t i = 0; i < m; ++i) {
        auto i_ = static_cast<Eigen::Index>(i);

        S(i, i) = S_(i_);
    }

    auto U = zeros<matrix_type::general>(queue, n, m, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            auto i_ = static_cast<Eigen::Index>(i);
            auto j_ = static_cast<Eigen::Index>(j);

            U(i, j) = U_(i_, j_);
        }
    }

    return std::make_tuple(std::move(U), std::move(S));
}
}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
