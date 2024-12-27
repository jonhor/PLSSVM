#ifndef PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"
#include <tuple>

namespace plssvm::sycl::linalg {

/**
 * Compute a thin SVD, such that A = U * Sigma * VT
 * We only need to compute U and Sigma for RPCholesky.
 */
using svd_return_type = std::tuple<matrix<matrix_type::general>, matrix<matrix_type::diagonal>>;

svd_return_type svd(::sycl::queue &queue, const matrix_view<matrix_type::general> &A);

}
#endif // PLSSVM_BACKENDS_SYCL_LINALG_SVD_HPP_
