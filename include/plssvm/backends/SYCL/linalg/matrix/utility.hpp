#ifndef PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_UTILITY_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_UTILITY_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"

#include <filesystem>
#include <fmt/format.h>

/**
 *  Utility functions that are mainly used to debug or unit test.
 */

namespace plssvm::sycl::linalg::utility {

namespace internal {

/**
 * This is a helper function that constructs a matrix (with padding) from memory where values are stored in row-major order without any padding.
 * Mainly used in unit tests when loading large matrices from binary files.
 */
template <matrix_type T>
[[nodiscard]] matrix<T> from_linear_host_memory(::sycl::queue &queue, real_type *data, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    auto A_ = linalg::zeros<T>(queue, n_rows, n_cols, padding);

    const auto size = A_->size();
    const auto N = A_->n_rows;
    const auto M = A_->n_cols;
    auto A = A_.view();

    ::sycl::buffer<real_type, 1> data_buffer(data, ::sycl::range(size));
    if constexpr (T == matrix_type::general) {
        ::sycl::nd_range<2> nd_range{ ::sycl::range(N, M), ::sycl::range(linalg::BLOCK_SIZE, linalg::BLOCK_SIZE) };

        auto event = queue.submit([&](::sycl::handler &cgh) {
            auto data_acc = data_buffer.get_access<::sycl::access_mode::read>(cgh);
            cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                const auto global_row = item.get_global_id(0);
                const auto global_col = item.get_global_id(1);
                const auto global_linear_id = (global_row * M) + global_col;

                if (global_row >= N || global_col >= M) {
                    return;
                }

                A(global_row, global_col) = data_acc[global_linear_id];
            });
        });
        event.wait();

    } else if constexpr (T == matrix_type::upper || T == matrix_type::symmetric) {
        // Optimize:
        // There are formulas to compute the row and column directly from a given linear index
        // allowing the usage of a parallel kernel.
        // This function is only used in unit tests so optimizing this is not a priority.
        auto event = queue.submit([&](::sycl::handler &cgh) {
            auto data_acc = data_buffer.get_access<::sycl::access_mode::read>(cgh);
            cgh.single_task([=]() {
                std::size_t cur_idx = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    for (std::size_t j = i; j < M; ++j) {
                        A(i, j) = data_acc[cur_idx++];
                    }
                }
            });
        });
        event.wait();
    } else if constexpr (T == matrix_type::lower) {
        /*
         * Optimize: see upper
         */
        auto event = queue.submit([&](::sycl::handler &cgh) {
            auto data_acc = data_buffer.get_access<::sycl::access_mode::read>(cgh);
            cgh.single_task([=]() {
                std::size_t cur_idx = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    for (std::size_t j = 0; j <= i; ++j) {
                        A(i, j) = data_acc[cur_idx++];
                    }
                }
            });
        });
        event.wait();
    } else if constexpr (T == matrix_type::diagonal) {
        ::sycl::nd_range<1> nd_range{ ::sycl::range(size), ::sycl::range(linalg::BLOCK_SIZE * linalg::BLOCK_SIZE) };

        auto event = queue.submit([&](::sycl::handler &cgh) {
            auto data_acc = data_buffer.get_access<::sycl::access_mode::read>(cgh);
            cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<1> &item) {
                const auto global_idx = item.get_global_id();

                if (global_idx >= size) {
                    return;
                }

                A(global_idx, global_idx) = data_acc[global_idx];
            });
        });
        event.wait();
    } else {
        static_assert(false);
    }

    return A_;
}

}  // namespace internal

template <matrix_type T>
[[nodiscard]] matrix<T> read_matrix_from_file(::sycl::queue &queue, const std::string &file_path, std::size_t n_rows, std::size_t n_cols, std::size_t padding = 0) {
    // All test data is currently stored in double format.
    // One could still read those values as doubles and then cast them to float.
    static_assert(std::is_same<real_type, double>::value, "this doesn't work correctly when not compiling with double");

    PLSSVM_ASSERT(std::filesystem::exists(file_path), fmt::format("file {} does not exist", file_path));
    PLSSVM_ASSERT(std::filesystem::is_regular_file(file_path), fmt::format("file {} is not a regular file", file_path));

    // open file
    std::streampos file_size;
    std::ifstream file(file_path, std::ios::binary);

    // get file size
    file.seekg(0, std::ios::end);
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const auto n_elements = static_cast<std::size_t>(file_size / static_cast<std::streamoff>(sizeof(double)));
    const auto matrix_size = linalg::internal::size<T>(n_rows, n_cols);
    PLSSVM_ASSERT(n_elements == matrix_size, fmt::format("number of elements in file does not equal the expected matrix size, {} != {}", n_elements, matrix_size));

    // read data
    // TODO use a single buffer and use host accessor here
    double *data = new double[n_elements];
    file.read((char *) data, file_size);

    auto A = internal::from_linear_host_memory<T>(queue, data, n_rows, n_cols, padding);
    delete[] data;

    return A;
}

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
    } else if constexpr (T == matrix_type::diagonal) {
        const auto n = std::min(A.n_rows, A.n_cols);
        for (std::size_t i = 0; i < n; ++i) {
            const auto value = A(i, i);
            if (std::isnan(value) || std::isinf(value)) {
                fmt::println("index ({}, {}) is not valid", i, i);
                return false;
            }
        }
    } else {
        PLSSVM_ASSERT(false, "not implemented");
    }

    return true;
}

template <matrix_type T>
void print(const matrix_view<T> &A) {
    // printing is only allowed in shared mode for now.
    static_assert(linalg::internal::alloc_mode == linalg::internal::allocation_mode::shared);

    if constexpr (T == matrix_type::general) {
        for (std::size_t i = 0; i < A.n_rows; ++i) {
            for (std::size_t j = 0; j < A.n_cols; ++j) {
                fmt::print("{:.4f} ", A(i, j));
            }
            fmt::println("");
        }
    } else if (T == matrix_type::diagonal) {
        const auto k = std::min(A.n_rows, A.n_cols);
        for (std::size_t i = 0; i < k; ++i) {
            fmt::print("{:.4f} ", A(i, i));
        }
        fmt::println("");
    }
}

}  // namespace plssvm::sycl::linalg::utility

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_MATRIX_UTILITY_HPP_
