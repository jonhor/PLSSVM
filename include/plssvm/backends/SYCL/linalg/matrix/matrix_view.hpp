#ifndef PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_VIEW_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_VIEW_HPP_
#pragma once

#include "plssvm/constants.hpp"      // plssvm::real_type
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/shape.hpp"          // plssvm::shape

#include "sycl/sycl.hpp"

#include "matrix_types.hpp"

namespace plssvm::sycl::linalg {

/*
 * A lightweight 2-dimensional matrix wrapper over a raw pointer.
 *
 * This type is and should remain trivially copyable because
 * it is supposed to be used in SYCL kernel functions.
 */
template <matrix_type T, bool transposed = false>
class matrix_view final {
  public:
    matrix_view(real_type *data, shape shape, std::size_t padding) :
        data_(data),
        shape(shape),
        n_rows(shape.x),
        n_cols(shape.y),
        padding(padding) {
        if constexpr (T != matrix_type::general) {
            PLSSVM_ASSERT(n_rows == n_cols, "triangular matrix should be symmetric");
        }

        // #if !defined(NDEBUG)
        //         padding_is_valid()
        // #endif
    }

    matrix_view(real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding) :
        matrix_view(data, plssvm::shape(n_rows, n_cols), padding) { }

    matrix_view(real_type *data, shape shape) :
        matrix_view(data, shape, 0) { }

    matrix_view(real_type *data, std::size_t n_rows, std::size_t n_cols) :
        matrix_view(data, plssvm::shape(n_rows, n_cols), 0) { }

    [[nodiscard]] inline real_type &operator()(std::size_t row, std::size_t col) const {
        if constexpr (transposed) {
            return data_[index(col, row)];
        }
        return data_[index(row, col)];
    }

    [[nodiscard]] std::size_t size() const {
        return internal::size<T>(shape.x, shape.y);
    }

    [[nodiscard]] std::size_t size_padded() const {
        return internal::size_padded<T>(shape.x, shape.y, padding);
    }

    [[nodiscard]] std::size_t size_bytes() const {
        return internal::size_bytes<T>(shape.x, shape.y);
    }

    [[nodiscard]] std::size_t size_bytes_padded() const {
        return internal::size_bytes_padded<T>(shape.x, shape.y, padding);
    }

    [[nodiscard]] real_type *data() const {
        return data_;
    }

  private:
    [[nodiscard]] inline constexpr std::size_t index(std::size_t row, std::size_t col) const {
        auto r = row;
        auto c = col;

        if constexpr (T == matrix_type::general) {
            return (row * n_cols + col) + (padding * row);
        }
        
        if constexpr (T == matrix_type::diagonal) {
            // if row == col -> return value
            // if row != col -> return 0
            // if oob access -> ?
            return row;
        }

        if constexpr (T == matrix_type::upper || T == matrix_type::symmetric) {
            // indexing is basically the same between those two because symmetric matrices are stored in upper triangular form.
            // indexing into the lower triangular
            //  -> returns 0 for upper triangular matrices.
            //  -> returns (col, row) for symmetric matrices.
            if (row > col) {
                r = col;
                c = row;
            }
            return ((r * (2 * n_rows - r + 1)) / 2) + (c - r) + (padding * r);
        }

        if constexpr (T == matrix_type::lower) {
            // if row >= col -> return value
            // if col > row -> return 0
            // if oob access -> ?
            return ((row * (row + 1)) / 2 + col) + (padding * row);
        }
    }

    real_type *data_;  // non-owned memory where matrix elements are stored in row-major order

  public:
    // storing both the shape and the number of rows and columns is redundant, nonetheless it is often times more
    // intuitive to refer to the dimensions of a matrix in terms of rows and columns instead of x and y
    const plssvm::shape shape;
    const std::size_t n_rows, n_cols;
    const std::size_t padding;
};
}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_VIEW_HPP_
