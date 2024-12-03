#ifndef PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_HPP_
#pragma once

#include "sycl/sycl.hpp"

#include "matrix_types.hpp"
#include "matrix_view.hpp"

namespace plssvm::sycl::linalg {

namespace internal {
// TODO connect this with the new cg mode
enum class allocation_mode {
    shared,
    device,
};
constexpr auto alloc_mode = allocation_mode::shared;

}  // namespace internal

/*
 * A simple RAII wrapper around matrix_view.
 *
 * This type is intended to be move-only.
 */
template <matrix_type T>
class matrix final {
  public:
    matrix(const ::sycl::queue &queue, const matrix_view<T> &view) :
        queue_(queue),
        view_(view) { }

    matrix(const ::sycl::queue &queue, const matrix_view<T> &&view) :
        queue_(queue),
        view_(view) { }

    matrix(const matrix &) = delete;
    matrix &operator=(const matrix &) = delete;

    matrix(matrix &&other) :
        queue_(std::move(other.queue_)),
        view_(std::move(other.view_)) {
        other.owns_data_ = false;
    }

    matrix &operator=(matrix &&) = delete;

    ~matrix() {
        if (owns_data_) {
            ::sycl::free(view_.data(), queue_);
        }
    }

    operator matrix_view<T> &() {
        return view_;
    }

    operator const matrix_view<T> &() const {
        return view_;
    }

    matrix_view<T> *operator->() {
        return &view_;
    }

    [[nodiscard]] inline real_type &operator()(std::size_t row, std::size_t col) const {
        return view_(row, col);
    }

    [[nodiscard]] matrix_view<T> &view() {
        return view_;
    }

    [[nodiscard]] const matrix_view<T> &view() const {
        return view_;
    }

    void release() {
        owns_data_ = false;
    }

  private:
    ::sycl::queue queue_;
    matrix_view<T> view_;
    bool owns_data_ = true;
};

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_HPP_
