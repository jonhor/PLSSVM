#ifndef PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_

#include "plssvm/backends/SYCL/detail/block.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"

#include "sycl/sycl.hpp"

namespace plssvm::sycl::linalg {

class cholesky_decomposition {
  public:
    cholesky_decomposition(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &A, std::size_t block_size = 0) :
        queue_(queue),
        A_(A),
        U_matrix(empty<matrix_type::upper>(queue, A.n_rows, A.n_rows, A.padding)),
        block_size_(block_size == 0 ? BLOCK_SIZE : block_size) {
    }

    inline matrix<matrix_type::upper> operator()() {
        /*
         * Copy A to U and then perform the decomposition on U inplace.
         * This copy is safe because a symmetric matrix has the same memory layout as an
         * upper triangular matrix.
         */
        queue_.memcpy(U_matrix->data(), A_.data(), A_.size_bytes_padded()).wait();

        /*
         * The loop consists of 3 main steps that are performed until the full decomposition is computed.
         * - compute the cholesky factorization of the current diagonal block
         * - solve the trailing blocks in the same row
         * - update the blocks in the trailing submatrix
         */
        std::chrono::steady_clock::time_point start_time, end_time;
        const auto N = A_.n_rows;
        for (std::size_t row_offset = 0; row_offset < N; row_offset += block_size_) {
            const auto remaining_block_size = std::min(block_size_, N - row_offset);

            start_time = std::chrono::steady_clock::now();
            factorize_block(row_offset, remaining_block_size).wait();
            end_time = std::chrono::steady_clock::now();
            total_factorization_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // solve all trailing blocks in the same row
            start_time = std::chrono::steady_clock::now();
            for (std::size_t col_offset = row_offset + block_size_; col_offset < N; col_offset += block_size_) {
                solve_block(row_offset, col_offset);
            }
            queue_.wait();
            end_time = std::chrono::steady_clock::now();
            total_solve_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

            // update the blocks in the trailing submatrix
            start_time = std::chrono::steady_clock::now();
            for (std::size_t trailing_row_offset = row_offset + block_size_; trailing_row_offset < N; trailing_row_offset += block_size_) {
                std::size_t blocks_in_row = 0;
                for (std::size_t col = trailing_row_offset; col < N; col += block_size_) {
                    blocks_in_row += 1;
                }
                update_trailing_block_row(trailing_row_offset, row_offset, blocks_in_row);
            }
            queue_.wait();
            end_time = std::chrono::steady_clock::now();
            total_update_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        }

        return std::move(U_matrix);
    }

    std::chrono::milliseconds total_factorization_time() const {
        return total_factorization_time_;
    }

    std::chrono::milliseconds total_solve_time() const {
        return total_solve_time_;
    }

    std::chrono::milliseconds total_update_time() const {
        return total_update_time_;
    }

  private:
    /**
     * Factorizes a single block row by row.
     *
     * This function essentially performs the same steps as the block-wise algorithm on single elements.
     * - compute diagonal element
     * - update elements in the same row
     * - update the trailing submatrix
     */
    inline ::sycl::event factorize_block(std::size_t row_offset, std::size_t block_size) {
        const ::sycl::range local_range(block_size, block_size);
        ::sycl::nd_range nd_range{ local_range, local_range };

        auto U_ = U_matrix.view();

        // use the smallest possible epsilon for the selected real type to stabilize the decomposition.
        auto smallest_eps = std::numeric_limits<real_type>::epsilon();

        return queue_.submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<real_type, 2> cache(local_range, cgh);

            cgh.parallel_for<class cholesky_factorize_block>(nd_range, [=](::sycl::nd_item<2> item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                const auto global_row = row + row_offset;
                const auto global_col = col + row_offset;

                // only work with the upper triangular part
                if (row > col) {
                    return;
                }

                cache[row][col] = U_(global_row, global_col);
                item.barrier(::sycl::access::fence_space::local_space);

                for (std::size_t current_row = 0; current_row < block_size; ++current_row) {
                    /*
                     * For the cholesky decomposition to work correctly the matrix is required to be positive semi-definite.
                     * In practice most algorithms require the matrix to be positive definite to avoid dividing by zero.
                     * While kernel matrices fulfill this requirement in theory, diagonal elements can still become negative
                     * due to numerical instability / inaccuracies.
                     *
                     * This is why we clip diagonal elements to be at least the smallest possible epsilon depending on the selected real type.
                     */

                    // only one thread accesses and updates the diagonal element of the current row.
                    if (row == 0 && col == 0) {
                        auto diag_element = std::max(cache[current_row][current_row], smallest_eps);
                        cache[current_row][current_row] = std::sqrt(diag_element);
                    }
                    item.barrier(::sycl::access::fence_space::local_space);

                    // update the rest of the row
                    if (row == current_row && col > current_row) {
                        cache[current_row][col] /= cache[current_row][current_row];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);

                    // update the trailing submatrix
                    if (row > current_row) {
                        cache[row][col] -= cache[current_row][row] * cache[current_row][col];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                U_(global_row, global_col) = cache[row][col];
            });
        });
    }

    /**
     * Performs a triangular solve on a block in the current row of blocks.
     */
    inline ::sycl::event solve_block(std::size_t row_offset, std::size_t col_offset) {
        const auto N = block_size_;
        const ::sycl::range local_range(N, N);
        const ::sycl::nd_range nd_range(local_range, local_range);

        auto U_ = U_matrix.view();
        return queue_.submit([&](::sycl::handler &cgh) {
            // cgh.depends_on(factorize_event);

            ::sycl::local_accessor<real_type, 2> b_cache(local_range, cgh);  // cache for the current block
            ::sycl::local_accessor<real_type, 2> d_cache(local_range, cgh);  // cache for the diagonal block in the current row

            cgh.parallel_for<class cholesky_solve_block>(nd_range, [=](::sycl::nd_item<2> item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);

                const auto global_row = row + row_offset;
                const auto global_col = col + col_offset;
                const auto diag_col = col + row_offset;

                b_cache[row][col] = U_(global_row, global_col);

                // transpose the data in the diagonal block to perform a forward substitution
                if (row <= col) {
                    d_cache[col][row] = U_(global_row, diag_col);
                }
                item.barrier(::sycl::access::fence_space::local_space);

                for (std::size_t current_row = 0; current_row < N; ++current_row) {
                    if (row == current_row) {
                        real_type sum = 0;
                        for (std::size_t k = 0; k < current_row; ++k) {
                            sum += d_cache[current_row][k] * b_cache[k][col];
                        }
                        b_cache[current_row][col] = (b_cache[current_row][col] - sum) / d_cache[current_row][current_row];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                U_(global_row, global_col) = b_cache[row][col];
            });
        });
    }

    /**
     * Updates a single row of blocks in the trailing submatrix.
     *
     * The approach of updating rows in parallel has the advantage that one of the blocks from the
     * currently solved row can be held in cache for all updated blocks in the same row.
     */
    inline ::sycl::event update_trailing_block_row(std::size_t row_offset, std::size_t solved_row_offset, std::size_t n_blocks) {
        const auto N = block_size_;

        const ::sycl::range<2> global_range(N, N * n_blocks);
        const ::sycl::range<2> local_range(N, N);
        const ::sycl::nd_range<2> nd_range(global_range, local_range);

        auto U_ = U_matrix.view();
        return queue_.submit([&](::sycl::handler &cgh) {
            const ::sycl::local_accessor<real_type, 2> i_cache(local_range, cgh);
            const ::sycl::local_accessor<real_type, 2> j_cache(local_range, cgh);

            cgh.parallel_for<class cholesky_update_trailing_block_row>(nd_range, [=](::sycl::nd_item<2> item) {
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);
                const auto global_row = row + row_offset;
                const auto global_col = item.get_global_id(1) + row_offset;

                // U_li ^ T @ U_lj, where l is the last solved row
                i_cache[col][row] = U_(row + solved_row_offset, row_offset + col);
                j_cache[row][col] = U_(row + solved_row_offset, global_col);
                item.barrier(::sycl::access::fence_space::local_space);

                real_type sum = 0;
                for (std::size_t k = 0; k < N; ++k) {
                    sum += i_cache[row][k] * j_cache[k][col];
                }

                if (global_row <= global_col) {
                    U_(global_row, global_col) -= sum;
                }
            });
        });
    }

    ::sycl::queue queue_;

    matrix_view<matrix_type::symmetric> A_;
    matrix<matrix_type::upper> U_matrix;
    // matrix_view<matrix_type::upper> U_;
    const std::size_t block_size_;

    std::chrono::milliseconds total_factorization_time_{ 0 };
    std::chrono::milliseconds total_solve_time_{ 0 };
    std::chrono::milliseconds total_update_time_{ 0 };
};

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_CHOLESKY_HPP_
