#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
#pragma once

#include "plssvm/constants.hpp"      // plssvm::real_type
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/shape.hpp"          // plssvm::shape

#include "random.hpp"
#include <initializer_list>  // std::initializer_list
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

namespace shared {

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size(std::size_t n_rows, std::size_t n_cols) {
    if constexpr (T == matrix_type::general) {
        return n_rows * n_cols;
    }
    return (n_rows * (n_rows + 1)) / 2;
}

template <matrix_type T>
[[nodiscard]] inline constexpr std::size_t size_padded(std::size_t n_rows, std::size_t n_cols, std::size_t padding) {
    if constexpr (T == matrix_type::general) {
        return (n_rows + padding) * (n_cols + padding);
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

//
// template <matrix_type T>
// [[nodiscard]] constexpr bool verify_padding(real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding) {
//     if (padding == 0) {
//         return true;
//     }
//     if constexpr (T == matrix_type::upper) {
//         const auto last_col = n_cols - 1;
//         for (std::size_t row = 0; row < n_rows; ++row) {
//         }
//
//     } else {
//         PLSSVM_ASSERT(false, "paddding verification is not implemented for this matrix type yet");
//     }
// }

}  // namespace shared

/*
 * A lightweight 2-dimensional matrix wrapper over a raw pointer
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
        return shared::size<T>(shape.x, shape.y);
    }

    [[nodiscard]] std::size_t size_padded() const {
        return shared::size_padded<T>(shape.x, shape.y, padding);
    }

    [[nodiscard]] std::size_t size_bytes() const {
        return shared::size_bytes<T>(shape.x, shape.y);
    }

    [[nodiscard]] std::size_t size_bytes_padded() const {
        return shared::size_bytes_padded<T>(shape.x, shape.y, padding);
    }

    [[nodiscard]] real_type *data() const {
        return data_;
    }

  private:
    [[nodiscard]] inline constexpr std::size_t index(std::size_t row, std::size_t col) const {
        if constexpr (T == matrix_type::general) {
            return (row * n_cols + col) + (padding * row);
        }
        if constexpr (T == matrix_type::lower) {
            return ((row * (row + 1)) / 2 + col) + (padding * row);
        }
        return ((row * (2 * n_rows - row + 1)) / 2) + (col - row) + (padding * row);
    }

    real_type *data_;  // non-owned memory where matrix elements are stored in row-major order

  public:
    // storing both the shape and the number of rows and columns is redundant, nonetheless it is often times more
    // intuitive to refer to the dimensions of a matrix in terms of rows and columns instead of x and y
    const plssvm::shape shape;
    const std::size_t n_rows, n_cols;
    const std::size_t padding;
};

/*
 * A simple RAII wrapper around matrix_view, for the rare case that matrix_view actually owns its memory
 */
template <matrix_type T>
class managed_matrix_view final {
  public:
    managed_matrix_view(const ::sycl::queue &queue, const matrix_view<T> &view) :
        queue_(queue),
        view_(view) { }

    managed_matrix_view(const ::sycl::queue &queue, const matrix_view<T> &&view) :
        queue_(queue),
        view_(view) { }

    managed_matrix_view(const managed_matrix_view &) = delete;
    managed_matrix_view &operator=(const managed_matrix_view &) = delete;

    managed_matrix_view(managed_matrix_view &&) = default;
    managed_matrix_view &operator=(managed_matrix_view &&) = delete;

    ~managed_matrix_view() {
        if (!released_) {
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
        released_ = true;
    }

  private:
    ::sycl::queue queue_;
    matrix_view<T> view_;
    bool released_ = false;
};

namespace utility {

template <matrix_type T>
bool is_valid(const matrix_view<T> &A) {
    if constexpr (T == matrix_type::upper) {
        for (std::size_t i = 0; i < A.n_rows; ++i) {
            for (std::size_t j = i; j < A.n_rows; ++j) {
                const auto value = A(i, j);
                if (std::isnan(value) || std::isinf(value)) {
                    fmt::println("index ({}, {}) is not valid", i, j);
                    return false;
                }
            }
        }
    } else if constexpr (T == matrix_type::general) {
        for (std::size_t i = 0; i < A.n_rows; ++i) {
            for (std::size_t j = 0; j < A.n_cols; ++j) {
                const auto value = A(i, j);
                if (std::isnan(value) || std::isinf(value)) {
                    fmt::println("index ({}, {}) is not valid", i, j);
                    return false;
                }
            }
        }
    } else {
        PLSSVM_ASSERT(false, "not implemented");
    }

    return true;
}

/**
 * Creates a managed view with the specified shape, where the memory is initialized to zero.
 */
template <matrix_type T>
managed_matrix_view<T> zeros(::sycl::queue &queue, const shape &shape, std::size_t padding = 0) {
    const auto size_bytes_padded = shared::size_bytes_padded<matrix_type::general>(shape.x, shape.y, padding);
    auto *view_data = ::sycl::malloc_shared<real_type>(size_bytes_padded, queue);
    queue.memset(view_data, 0, size_bytes_padded).wait();

    return managed_matrix_view(queue, matrix_view<T>(view_data, shape, padding));
}

template <matrix_type T>
managed_matrix_view<T> zeros(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    return zeros<T>(queue, shape(n_rows, n_cols), padding);
}

template <matrix_type T>
managed_matrix_view<T> create_managed_view(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    const auto size_bytes_padded = shared::size_bytes_padded<T>(n_rows, n_cols, padding);
    auto *view_data = ::sycl::malloc_shared<real_type>(size_bytes_padded, queue);

    return managed_matrix_view(queue, matrix_view<T>(view_data, n_rows, n_cols, padding));
}

template <matrix_type T>
managed_matrix_view<T> create_managed_view(::sycl::queue &queue, real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A = zeros<T>(queue, n_rows, n_cols, padding);

    std::size_t idx = 0;
    if constexpr (T == matrix_type::general) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = 0; j < n_cols; ++j) {
                A(i, j) = data[idx];
                idx++;
            }
        }
    } else if constexpr (T == matrix_type::upper) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = i; j < n_cols; ++j) {
                A(i, j) = data[idx];
                idx++;
            }
        }
    } else if constexpr (T == matrix_type::lower) {
        for (std::size_t row = 0; row < n_rows; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                A(row, col) = data[idx];
                idx++;
            }
        }
    }

    return A;
}

/**
 * Creates a managed view over shared memory allocated using SYCL.
 * This initializer_list version should only be used for unit tests and never in actual code.
 */
template <matrix_type T>
managed_matrix_view<T> create_managed_view(::sycl::queue &queue, std::initializer_list<real_type> elems, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A = zeros<T>(queue, n_rows, n_cols, padding);
    PLSSVM_ASSERT(elems.size() == A->size(), fmt::format("initializer list should have the same size as the provided shape, {} != {}", elems.size(), A->size()));

    auto it = elems.begin();
    if constexpr (T == matrix_type::general) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = 0; j < n_cols; ++j) {
                A(i, j) = *it;
                it++;
            }
        }
    } else if constexpr (T == matrix_type::upper) {
        for (std::size_t i = 0; i < n_rows; ++i) {
            for (std::size_t j = i; j < n_cols; ++j) {
                A(i, j) = *it;
                it++;
            }
        }
    } else if constexpr (T == matrix_type::lower) {
        for (std::size_t row = 0; row < n_rows; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                A(row, col) = *it;
                it++;
            }
        }
    }

    return A;
}

/**
 * Creates a managed view by allocating memory and copying elements from data.
 */
template <matrix_type T>
managed_matrix_view<T> create_managed_view(::sycl::queue &queue, const real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A = zeros<T>(queue, shape(n_rows, n_cols), padding);
    queue.memcpy(A->data(), data, A->size_bytes_padded()).wait_and_throw();

    return A;
}

/**
 * Transpose by allocating new memory and copying the elements in parallel.
 */
inline managed_matrix_view<matrix_type::lower> transpose(::sycl::queue &queue, const matrix_view<matrix_type::upper> &U) {
    auto L_ = zeros<matrix_type::lower>(queue, U.shape, U.padding);
    auto L = L_.view();

    const auto block_size = PADDING_SIZE;

    ::sycl::range<2> global_range(U.n_rows, U.n_rows);
    ::sycl::range<2> local_range(block_size, block_size);
    ::sycl::nd_range<2> nd_range(global_range, local_range);

    queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
            const auto global_row = item.get_global_id(0);
            const auto global_col = item.get_global_id(1);

            if (global_col >= global_row) {
                L(global_col, global_row) = U(global_row, global_col);
            }
        });
    });
    queue.wait();

    return L_;
}

//
// inline managed_matrix_view<matrix_type::upper> transpose(::sycl::queue &queue, const matrix_view<matrix_type::lower> &L) {
//    auto U_ = zeros<matrix_type::upper>(queue, L.shape, L.padding);
//    auto U = U_.view();
//
//    ::sycl::range<2> range_xy(L.n_rows, L.n_rows);
//    queue.submit([&](::sycl::handler &cgh) {
//        cgh.parallel_for(range_xy, [=](::sycl::id<2> idx) {
//            const auto row = idx[0];
//            const auto col = idx[1];
//
//            if (row >= col) {
//                U(col, row) = L(row, col);
//            }
//        });
//    });
//    queue.wait();
//
//    return U_;
//}

template <matrix_type T>
managed_matrix_view<T> randn(::sycl::queue &queue, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A = create_managed_view<T>(queue, n_rows, n_cols, padding);

    rng rand{};
    for (std::size_t i = 0; i < n_rows; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            A(i, j) = rand.randf();
        }
    }

    return A;
}

void dump_view_to_file(const matrix_view<matrix_type::upper> &A, const char *file_name) {
    // open file
    std::ofstream file(file_name, std::ios::binary);

    const auto N = A.n_rows;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i; j < N; ++j) {
            file.write((char *) &A(i, j), sizeof(real_type));
        }
    }
}

std::vector<double> read_data_from_file(const char *file_name) {
    // open file
    std::streampos file_size;
    std::ifstream file(file_name, std::ios::binary);

    // get file size
    file.seekg(0, std::ios::end);
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // read data
    std::vector<double> file_data(file_size / sizeof(double));
    file.read((char *) &file_data[0], file_size);
    return file_data;
}

}  // namespace utility

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_MATRIX_VIEW_HPP
