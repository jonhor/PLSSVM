#ifndef PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_COMMON_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_COMMON_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/random.hpp"
#include "plssvm/backends/SYCL/detail/utility.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"
#include "plssvm/matrix.hpp"

#include "sycl/sycl.hpp"

#include <cstddef>

/**
 * A collection of common matrix-related functions.
 */

namespace plssvm::sycl::linalg {

namespace internal {

/**
 * Allocates memory using SYCLs USM model.
 */
inline real_type *allocate_memory(::sycl::queue &queue, std::size_t size_bytes) {
    if constexpr (alloc_mode == allocation_mode::shared) {
        return ::sycl::malloc_shared<real_type>(size_bytes, queue);
    }
    return ::sycl::malloc_device<real_type>(size_bytes, queue);
}

}  // namespace internal

template <matrix_type T>
[[nodiscard]] matrix<T> empty(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    const auto size_bytes_padded = internal::size_bytes_padded<matrix_type::general>(n_rows, n_cols, padding);
    auto *data = internal::allocate_memory(queue, size_bytes_padded);

    return matrix(queue, matrix_view<T>(data, n_rows, n_cols, padding));
}

template <matrix_type T>
[[nodiscard]] matrix<T> zeros(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A = empty<T>(queue, n_rows, n_cols, padding);
    queue.memset(A->data(), 0, A->size_bytes_padded()).wait();
    return A;
}

/**
 * Copy data from a plssvm::matrix which is stored on the host to a plssvm::sycl::linalg::matrix, where the memory is either shared or allocated on device.
 */
inline void copy_from_host_matrix(::sycl::queue &queue, matrix_view<matrix_type::general> &A, const soa_matrix<real_type> &H) {
    PLSSVM_ASSERT(H.padding().x == H.padding().y, "padding is to be expected symmetric");
    [[maybe_unused]] const auto H_size_bytes_padded = internal::size_bytes_padded<matrix_type::general>(H.num_rows(), H.num_cols(), H.padding().x);
    PLSSVM_ASSERT(H_size_bytes_padded == A.size_bytes_padded(), "A and H do not have the same size");

    // auto B = empty<matrix_type::general>(queue, A.num_rows(), A.num_cols(), A.padding().x);
    queue.memcpy(A.data(), H.data(), A.size_bytes_padded()).wait();
}

inline void copy_to_host_matrix(::sycl::queue &queue, const matrix_view<matrix_type::general> &A, soa_matrix<real_type> &H) {
    queue.memcpy(H.data(), A.data(), A.size_bytes_padded()).wait();
}

/**
 * Creates a matrix with the specified shape initialized with random integers in the specified distribution range.
 *
 * This function is only used in unit tests.
 */
template <typename IntType = int>
inline matrix<matrix_type::general> randint(::sycl::queue &queue, IntType begin, IntType end, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    PLSSVM_ASSERT(internal::alloc_mode == internal::allocation_mode::shared, "randint only works with shared allocation mode.");

    auto A = empty<matrix_type::general>(queue, n_rows, n_cols, padding);

    detail::rng rng{};
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            A(i, j) = static_cast<real_type>(rng.randint(begin, end));
        }
    }

    return A;
}

inline matrix<matrix_type::general> identity(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    PLSSVM_ASSERT(internal::alloc_mode == internal::allocation_mode::shared, "identity only works with shared allocation mode.");

    auto A = empty<matrix_type::general>(queue, n_rows, n_cols, padding);

    for (std::size_t i = 0; i < n_rows; ++i) {
        A(i, i) = real_type{ 1 };
    }
    return A;
}

/**
 * Extracts the diagonal from a matrix by allocating new memory and copying the diagonal elements.
 */
template <matrix_type T>
[[nodiscard]] matrix<matrix_type::diagonal> diagonal(::sycl::queue &queue, const matrix_view<T> &A) {
    auto D_ = zeros<matrix_type::diagonal>(queue, A.n_cols, A.n_rows, A.padding);

    const auto n_elements = std::min(A.n_rows, A.n_cols);

    auto D = D_.view();

    auto nd_range = detail::get_uniform_1d_range(n_elements, MAX_WORKGROUP_SIZE);
    auto event = queue.parallel_for<class copy_diagonal_elements>(nd_range, [=](::sycl::nd_item<1> item) {
        const auto global_id = item.get_global_id();

        if (global_id < n_elements) {
            D(global_id, global_id) = A(global_id, global_id);
        }
    });
    event.wait();

    return D_;
}

/**
 * Returns a transposed copy of A.
 */
[[nodiscard]] inline matrix<matrix_type::general> transposed(::sycl::queue &queue, const matrix_view<matrix_type::general> &A) {
    auto B_ = zeros<matrix_type::general>(queue, A.n_cols, A.n_rows, A.padding);
    auto B = B_.view();

    ::sycl::range<2> global_range(A.n_rows, A.n_cols);
    ::sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);
    ::sycl::nd_range<2> nd_range(global_range, local_range);

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<class general_transpose>(nd_range, [=](::sycl::nd_item<2> item) {
            const auto global_row = item.get_global_id(0);
            const auto global_col = item.get_global_id(1);

            B(global_col, global_row) = A(global_row, global_col);
        });
    });
    event.wait();

    return B_;
}

[[nodiscard]] inline matrix<matrix_type::lower> transposed(::sycl::queue &queue, const matrix_view<matrix_type::upper> &U) {
    auto L_ = zeros<matrix_type::lower>(queue, U.n_cols, U.n_rows, U.padding);
    auto L = L_.view();

    /*
     * Optimization:
     * We partition the matrix U here as we would a general NxN matrix,
     * but in practice we only need to consider the number of blocks equal to the upper triangular part of U.
     */
    ::sycl::nd_range nd_range{ ::sycl::range(U.n_rows, U.n_cols), ::sycl::range(BLOCK_SIZE, BLOCK_SIZE) };

    auto event = queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<class upper_transpose>(nd_range, [=](::sycl::nd_item<2> item) {
            const auto global_row = item.get_global_id(0);
            const auto global_col = item.get_global_id(1);

            if (global_col >= global_row) {
                L(global_col, global_row) = U(global_row, global_col);
            }
        });
    });
    event.wait();

    return L_;
}

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_COMMON_HPP_
