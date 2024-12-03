#ifndef PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_INVERSE_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_INVERSE_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix_types.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix_view.hpp"

namespace plssvm::sycl::linalg {

/**
 * Invert a diagonal matrix inplace.
 *
 * This operation will fail if there are zero elements on the diagonal.
 * Most kernel matrices are regularized with a cost factor C so this should not happen.
 */
inline void invert(::sycl::queue &queue, matrix_view<matrix_type::diagonal> &D) {
    const auto n_elements = std::min(D.n_rows, D.n_cols);

    ::sycl::nd_range<1> nd_range(::sycl::range(n_elements), ::sycl::range(BLOCK_SIZE * BLOCK_SIZE));
    auto event = queue.parallel_for<class invert_diagonal_matrix>(nd_range, [=](const ::sycl::nd_item<1> &item) {
        const auto global_id = item.get_global_id();

        D(global_id, global_id) = real_type{ 1 } / D(global_id, global_id);
    });
    event.wait();
}

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_INVERSE_HPP_
