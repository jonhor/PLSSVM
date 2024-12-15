#ifndef PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_

#include "plssvm/backends/SYCL/detail/block.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"

namespace plssvm::sycl::linalg {

// TODO comments about stability and stuff

namespace block {

inline void solve_triangular_lower(::sycl::queue &queue, const matrix_view<matrix_type::lower> &A, matrix_view<matrix_type::general> &B, std::size_t offset, std::size_t rows_to_solve, std::size_t block_size) {
    ::sycl::nd_range nd_range{ ::sycl::range(rows_to_solve, block_size), ::sycl::range(rows_to_solve, block_size) };

    auto solve_block = queue.submit([&](::sycl::handler &cgh) {
        ::sycl::range<2> solve_range(rows_to_solve, block_size);
        ::sycl::nd_range execution_range(solve_range, solve_range);

        ::sycl::local_accessor<real_type, 2> a_cache(solve_range, cgh);  // diagonal block from A
        ::sycl::local_accessor<real_type, 2> b_cache(solve_range, cgh);

        cgh.parallel_for<class trsm_solve_lower_block>(execution_range, [=](const ::sycl::nd_item<2> &item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);
            // we only have 1 work-group
            const auto global_row = row + offset;
            const auto global_col = col + offset;

            // load only the lower triangular part
            if (row >= col) {
                a_cache[row][col] = A(global_row, global_col);
            }
            b_cache[row][col] = B(global_row, col);
            item.barrier(::sycl::access::fence_space::local_space);

            for (std::size_t current_row = 0; current_row < rows_to_solve; ++current_row) {
                // solve the current row
                if (row == current_row) {
                    // std::max(eps, a_cache[current_row][current_row]);
                    b_cache[current_row][col] /= a_cache[current_row][current_row];
                }
                item.barrier(::sycl::access::fence_space::local_space);

                // update all trailing rows
                if (row > current_row) {
                    b_cache[row][col] -= a_cache[row][current_row] * b_cache[current_row][col];
                }
                item.barrier(::sycl::access::fence_space::local_space);
            }

            B(global_row, col) = b_cache[row][col];
        });
    });
    solve_block.wait();
}

inline void solve_triangular_upper(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::general> &B, std::size_t offset, std::size_t rows_to_solve, std::size_t block_size) {
    ::sycl::nd_range nd_range{ ::sycl::range(rows_to_solve, block_size), ::sycl::range(rows_to_solve, block_size) };

    auto solve_block = queue.submit([&](::sycl::handler &cgh) {
        ::sycl::range<2> solve_range(rows_to_solve, block_size);
        ::sycl::nd_range execution_range(solve_range, solve_range);

        ::sycl::local_accessor<real_type, 2> a_cache(solve_range, cgh);  // diagonal block from A
        ::sycl::local_accessor<real_type, 2> b_cache(solve_range, cgh);

        cgh.parallel_for<class trsm_solve_upper_block>(execution_range, [=](const ::sycl::nd_item<2> &item) {
            const auto row = item.get_local_id(0);
            const auto col = item.get_local_id(1);

            // we only have 1 work-group
            const auto global_row = row + offset;
            const auto global_col = col + offset;

            // load only the upper triangular part
            if (col >= row) {
                a_cache[row][col] = A(global_row, global_col);
            }
            b_cache[row][col] = B(global_row, col);
            item.barrier(::sycl::access::fence_space::local_space);

            // because we use a descending counter we have to be careful with signedness here.
            auto rows_to_solve_signed = static_cast<long>(rows_to_solve);
            for (long current_row_signed = rows_to_solve_signed - 1; current_row_signed >= 0; --current_row_signed) {
                auto current_row = static_cast<std::size_t>(current_row_signed);

                // solve the current row
                if (row == current_row) {
                    // std::max(eps, a_cache[current_row][current_row]);
                    b_cache[current_row][col] /= a_cache[current_row][current_row];
                }
                item.barrier(::sycl::access::fence_space::local_space);

                // update all rows above the current row
                if (row < current_row) {
                    b_cache[row][col] -= a_cache[row][current_row] * b_cache[current_row][col];
                }
                item.barrier(::sycl::access::fence_space::local_space);
            }

            B(global_row, col) = b_cache[row][col];
        });
    });
    solve_block.wait();
}

}  // namespace block

/**
 * Solves the linear system A * X = B for X, where A is a lower triangular matrix.
 * B is updated in place and will contain X at the end.
 */
inline void triangular_solve_lower_gpu(::sycl::queue &queue, const matrix_view<matrix_type::lower> &A, matrix_view<matrix_type::general> &B, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.n_rows == B.n_rows, "Order of A should be equal to the number of rows in B and X");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    // triangular solves are currently only supported if we can fit all columns of B/X into a block.
    // but it shouldn't be too difficult to adapt this algorithm to adapt.
    PLSSVM_ASSERT(B.n_cols <= block_size, "The number of columns of B should be smaller or equal to the block size");

    const auto N = B.n_rows;

    /*
     * This algorithm loops over all blocks in X, solving a block and then updating all remaining blocks.
     */
    for (std::size_t offset = 0; offset < N; offset += block_size) {
        const auto remaining_rows = N - offset;
        const auto rows_to_solve = std::min(block_size, remaining_rows);
        block::solve_triangular_lower(queue, A, B, offset, rows_to_solve, block_size);

        // update the rest of the blocks
        auto update_blocks = queue.submit([&](::sycl::handler &cgh) {
            ::sycl::range<2> local_range(block_size, block_size);
            ::sycl::range<2> global_range(remaining_rows, block_size);
            ::sycl::nd_range<2> execution_range(global_range, local_range);

            ::sycl::local_accessor<real_type, 2> a_cache(local_range, cgh);  // diagonal block from A that is multiplied with this block
            ::sycl::local_accessor<real_type, 2> b_cache(local_range, cgh);  // block from B that was last solved

            cgh.parallel_for<class trsm_update_blocks>(execution_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                auto global_row = item.get_global_id(0) + offset + block_size;

                a_cache[row][col] = A(global_row, col + offset);
                b_cache[row][col] = B(row + offset, col);
                item.barrier(::sycl::access::fence_space::local_space);

                real_type sum = 0;
                for (std::size_t k = 0; k < block_size; ++k) {
                    sum += a_cache[row][k] * b_cache[k][col];
                }

                B(global_row, col) -= sum;
            });
        });
        update_blocks.wait();
    }
}

inline void triangular_solve_upper_gpu(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, matrix_view<matrix_type::general> &B, std::size_t block_size = 0) {
    PLSSVM_ASSERT(A.n_rows == B.n_rows, "Order of A should be equal to the number of rows in B and X");

    block_size = block_size == 0 ? BLOCK_SIZE : block_size;
    // triangular solves are currently only supported if we can fit all columns of B/X into a block.
    // but it shouldn't be too difficult to adapt this algorithm to adapt.
    PLSSVM_ASSERT(B.n_cols <= block_size, "The number of columns of B should be smaller or equal to the block size");

    const auto N = B.n_rows;

    /*
     * This algorithm loops over all blocks in X, solving a block and then updating all remaining blocks.
     */
    const auto total_blocks = static_cast<std::size_t>(std::ceil(static_cast<double>(N) / static_cast<double>(block_size)));

    // because we use a descending counter we have to be careful with signedness here.
    const auto offset_start = static_cast<long>((total_blocks - 1) * block_size);
    for (long offset_signed = offset_start; offset_signed >= 0; offset_signed -= block_size) {
        auto offset = static_cast<std::size_t>(offset_signed);

        const auto remaining_rows = N - offset;
        const auto rows_to_solve = std::min(block_size, remaining_rows);
        block::solve_triangular_upper(queue, A, B, offset, rows_to_solve, block_size);

        if (offset == 0) {
            break;
        }

        // update the rest of the blocks
        auto update_blocks = queue.submit([&](::sycl::handler &cgh) {
            ::sycl::range<2> local_range(block_size, block_size);
            ::sycl::range<2> global_range(offset, block_size);
            ::sycl::nd_range<2> execution_range(global_range, local_range);

            ::sycl::local_accessor<real_type, 2> a_cache(local_range, cgh);  // diagonal block from A that is multiplied with this block
            ::sycl::local_accessor<real_type, 2> b_cache(local_range, cgh);  // block from B that was last solved

            cgh.parallel_for<class trsm_update_blocks>(execution_range, [=](const ::sycl::nd_item<2> &item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                auto global_row = item.get_global_id(0);

                a_cache[row][col] = A(global_row, col + offset);
                b_cache[row][col] = B(row + offset, col);
                item.barrier(::sycl::access::fence_space::local_space);

                real_type sum = 0;
                for (std::size_t k = 0; k < block_size; ++k) {
                    sum += a_cache[row][k] * b_cache[k][col];
                }

                B(global_row, col) -= sum;
            });
        });
        update_blocks.wait();
    }
}

template <matrix_type T>
matrix<matrix_type::general> triangular_solve_gpu(::sycl::queue &queue, const matrix_view<T> &A, const matrix_view<matrix_type::general> &B) {
    auto X = linalg::zeros<matrix_type::general>(queue, B.n_rows, B.n_cols, B.padding);
    queue.memcpy(X->data(), B.data(), B.size_bytes_padded()).wait();

    if constexpr (T == matrix_type::upper) {
        triangular_solve_upper_gpu(queue, A, X.view());
    } else if constexpr (T == matrix_type::lower) {
        triangular_solve_lower_gpu(queue, A, X.view());
    } else {
        static_assert(false);
    }

    return X;
}

inline void triangular_solve_lower(::sycl::queue &queue, const matrix_view<matrix_type::lower> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto n = A.n_rows;
    const auto m = B.n_cols;

    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t row = 0; row < n; ++row) {
            real_type sum = 0;
            for (std::size_t k = 0; k < row; ++k) {
                sum += A(row, k) * X(k, col);
            }

            X(row, col) = (X(row, col) - sum) / A(row, row);
        }
    }
}

inline void triangular_solve_upper(::sycl::queue &queue, const matrix_view<matrix_type::upper> &A, const matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &X) {
    queue.memcpy(X.data(), B.data(), B.size_bytes_padded()).wait();

    const auto n = A.n_rows;
    const auto m = B.n_cols;

    for (std::size_t col = 0; col < m; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto current_idx = n - i - 1;

            real_type sum = 0;
            for (std::size_t j = 0; j < i; ++j) {
                const auto current_j = n - j - 1;
                sum += A(current_idx, current_j) * X(current_j, col);
            }

            X(current_idx, col) = (X(current_idx, col) - sum) / A(current_idx, current_idx);
        }
    }
}

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_OPERATIONS_TRIANGULAR_SOLVE_HPP_
