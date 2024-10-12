#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type
// #include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include <initializer_list>
#include <sycl/sycl.hpp>

namespace plssvm::sycl::detail {

enum class matrix_type {
    /* general (n x k) matrix */
    general,
    /* lower triangular matrix */
    lower,
    /* upper triangular matrix */
    upper
};

/*
 * A lightweight 2-dimensional matrix wrapper over a raw pointer
 */
template <matrix_type type>
class matrix_view {
  public:
    matrix_view(real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding) :
        data_(data),
        n_rows(n_rows),
        n_cols(n_cols),
        padding(padding) { }

    matrix_view(real_type *data, std::size_t n_rows, std::size_t n_cols) :
        matrix_view(data, n_rows, n_cols, 0) { }

    matrix_view(real_type *data, std::size_t order) :
        matrix_view(data, order, order, 0) { }

    real_type &
    operator()(std::size_t row, std::size_t col) const;

    std::size_t size() const {
        return (n_rows * (n_rows + 1)) / 2;
    }

    std::size_t size_bytes() const {
        return size() * sizeof(real_type);
    }

    real_type *data() const {
        return data_;
    }

  private:
    real_type *data_;  // non-owned memory where matrix elements are stored in row-major order
  public:
    std::size_t n_rows, n_cols;  // number of rows and columns
    std::size_t padding;
};

template <>
inline std::size_t matrix_view<matrix_type::general>::size() const {
    return n_rows * n_cols;
}

template <>
inline real_type &matrix_view<matrix_type::general>::operator()(std::size_t row, std::size_t col) const {
    return data_[(row * n_cols + col) + (padding * row)];
}

template <>
inline real_type &matrix_view<matrix_type::lower>::operator()(std::size_t row, std::size_t col) const {
    // PLSSVM_ASSERT(row >= col, "row should always be greater or equal to col when accessing a lower triangular matrix");
    return data_[((row * (row + 1)) / 2 + col) + (padding * row)];
}

template <>
inline real_type &matrix_view<matrix_type::upper>::operator()(std::size_t row, std::size_t col) const {
    // PLSSVM_ASSERT(col >= row, "col should always be greater or equal to row when accessing an upper triangular matrix");
    auto idx = static_cast<std::size_t>((row * (2 * n_rows - row + 1)) / 2) + (col - row);
    idx += padding * row;
    return data_[idx];
}

namespace helper {

/*
 * TODO write tests
 * change shared to device
 */
inline matrix_view<matrix_type::general> create_shared_view(const real_type *data, size_t n_rows, size_t n_cols, size_t padding, ::sycl::queue &queue) {
    auto size_bytes = n_rows * n_cols * sizeof(real_type);
    auto *view_data = ::sycl::malloc_shared<real_type>(size_bytes, queue);
    queue.memcpy(view_data, data, size_bytes).wait();

    return matrix_view<matrix_type::general>(view_data, n_rows, n_cols, padding);
}

template <matrix_type T>
inline matrix_view<T> create_shared_view(std::initializer_list<real_type> elements, size_t n_rows, size_t n_cols, ::sycl::queue &queue) {
    auto size_bytes = elements.size() * sizeof(real_type);
    auto *view_data = ::sycl::malloc_shared<real_type>(size_bytes, queue);
    queue.memcpy(view_data, data(elements), size_bytes).wait();

    return matrix_view<T>(view_data, n_rows, n_cols);
}

template <matrix_type T>
inline matrix_view<T> zeros_like(const matrix_view<T> &A, ::sycl::queue &queue) {
    auto *view_data = ::sycl::malloc_shared<real_type>(A.size_bytes(), queue);
    queue.memset(view_data, 0, A.size_bytes()).wait();

    return matrix_view<T>(view_data, A.n_rows, A.n_cols, A.padding);
}

/*
 * Transpose by allocating new memory and copying the elements in parallel.
 */
inline matrix_view<matrix_type::lower> transpose(const matrix_view<matrix_type::upper> &U, ::sycl::queue &queue) {
    auto *view_data = ::sycl::malloc_shared<real_type>(U.size_bytes(), queue);
    matrix_view<matrix_type::lower> L(view_data, U.n_rows);

    ::sycl::range<2> range_xy(U.n_rows, U.n_rows);
    queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(range_xy, [=](::sycl::id<2> idx) {
            const auto row = idx[0];
            const auto col = idx[1];

            if (col >= row) {
                L(col, row) = U(row, col);
            }
        });
    });
    queue.wait();

    return L;
}

inline matrix_view<matrix_type::upper> transpose(const matrix_view<matrix_type::lower> &L, ::sycl::queue &queue) {
    auto *view_data = ::sycl::malloc_shared<real_type>(L.size_bytes(), queue);
    matrix_view<matrix_type::upper> U(view_data, L.n_rows);

    ::sycl::range<2> range_xy(L.n_rows, L.n_rows);
    queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(range_xy, [=](::sycl::id<2> idx) {
            const auto row = idx[0];
            const auto col = idx[1];

            if (row >= col) {
                U(col, row) = L(row, col);
            }
        });
    });
    queue.wait();

    return U;
}

}  // namespace helper

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
