#ifndef PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_TYPES_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_TYPES_HPP_
#pragma once

#include "plssvm/constants.hpp"

#include <cstddef>

namespace plssvm::sycl::linalg {

// TODO write more here
enum class matrix_type {
    /* general (n x k) matrix */
    general,
    /* symmetric matrix, where only the upper triangular part is stored */
    symmetric,
    /* lower triangular matrix */
    lower,
    /* upper triangular matrix */
    upper,
    /* diagonal matrix */
    diagonal
};

namespace internal {

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size(std::size_t n_rows, std::size_t n_cols) {
    if constexpr (T == matrix_type::general) {
        return n_rows * n_cols;
    } else if constexpr (T == matrix_type::diagonal) {
        return std::min(n_rows, n_cols);
    }
    return (n_rows * (n_rows + 1)) / 2;
}

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size_padded(std::size_t n_rows, std::size_t n_cols, std::size_t padding) {
    if constexpr (T == matrix_type::general) {
        return (n_rows + padding) * (n_cols + padding);
    } else if constexpr (T == matrix_type::diagonal) {
        return size<T>(n_rows, n_cols) + padding;
    }
    return ((n_rows + padding) * (n_rows + padding + 1)) / 2;
}

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size_bytes(std::size_t n_rows, std::size_t n_cols) {
    return size<T>(n_rows, n_cols) * sizeof(real_type);
}

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size_bytes_padded(std::size_t n_rows, std::size_t n_cols, std::size_t padding) {
    return size_padded<T>(n_rows, n_cols, padding) * sizeof(real_type);
}

}  // namespace internal

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_TYPES_HPP_
