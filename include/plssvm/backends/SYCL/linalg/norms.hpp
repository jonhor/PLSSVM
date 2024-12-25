#ifndef PLSSVM_BACKENDS_SYCL_LINALG_NORMS_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_NORMS_HPP_

#include "plssvm/backends/SYCL/linalg/matrix/matrix_types.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix_view.hpp"

#include "sycl/sycl.hpp"

#include "constants.hpp"
#include <algorithm>
#include <execution>

namespace plssvm::sycl::linalg {

namespace kernels {
//
// inline void frobenius(::sycl::local_accessor<real_type, 2> cache, const std::size_t row, const std::size_t col, const ::sycl::nd_item<2> &item) {
//    // perform local reduction across columns
//    // such that the sum for each row is stored in the first column
//    for (auto offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
//        if (col < offset) {
//            cache[row][col] += cache[row][col + offset];
//        }
//        item.barrier(::sycl::access::fence_space::local_space);
//    }
//
//    // reduce across elements in the first column to compute the partial sum for this block
//    // which is then stored in the first element (0, 0) of the block
//    for (auto offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
//        if (col == 0 && row < offset) {
//            cache[row][col] += cache[row + offset][col];
//        }
//        item.barrier(::sycl::access::fence_space::local_space);
//    }
//
//    if (row == 0 && col == 0) {
//        // partial_sum_acc[item.get_group_linear_id()] = cache[0][0];
//    }
//}
//}  // namespace kernels
//
// inline real_type frobenius(::sycl::queue &queue, matrix_view<matrix_type::general> &A) {
//}

}  // namespace kernels

inline real_type frobenius(const matrix_view<matrix_type::general> &A, bool full = true) {
    const auto N = A.n_rows;
    const auto M = A.n_cols;

    real_type sum = real_type{ 0 };
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            sum += A(i, j) * A(i, j);
        }
    }

    return full ? std::sqrt(sum) : sum;
}

class frobenius_gpu {
  public:
    /*
     * Calculates the frobenius norm for the whole matrix A.
     */
    frobenius_gpu(::sycl::queue &queue, const matrix_view<matrix_type::general> &A) :
        queue_(queue),
        A_(A) {
    }

    real_type operator()(bool full = true) {
        PLSSVM_ASSERT(A_.padding >= BLOCK_SIZE, "padding should be at least block size");

        const auto N = A_.n_rows;
        const auto M = A_.n_cols;

        ::sycl::range<2> global_range(N, M);
        ::sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);
        ::sycl::nd_range<2> nd_range(global_range, local_range);

        const auto num_blocks = static_cast<std::size_t>(
            std::ceil(static_cast<double>(N) / static_cast<double>(BLOCK_SIZE)) * std::ceil(static_cast<double>(M) / static_cast<double>(BLOCK_SIZE)));

        ::sycl::buffer<real_type> partial_sums{ ::sycl::range<1>{ num_blocks } };

        auto sum_event = queue_.submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<real_type, 2> local_cache(local_range, cgh);
            auto partial_sum_acc = partial_sums.get_access<::sycl::access_mode::discard_write>(cgh);

            cgh.parallel_for(nd_range, [=](const ::sycl::nd_item<2> &item) {
                const auto global_row = item.get_global_id(0);
                const auto global_col = item.get_global_id(1);
                const auto row = item.get_local_id(0);
                const auto col = item.get_local_id(1);

                local_cache[row][col] = ::sycl::pow(A_(global_row, global_col), 2);
                item.barrier(::sycl::access::fence_space::local_space);

                // perform local reduction across columns
                // such that the sum for each row is stored in the first column
                for (auto offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                    if (col < offset) {
                        local_cache[row][col] += local_cache[row][col + offset];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                // reduce across those row elements to compute the partial sum for this block
                // which is then stored in the first element (0, 0) of the block
                for (auto offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                    if (col == 0 && row < offset) {
                        local_cache[row][col] += local_cache[row + offset][col];
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }

                if (row == 0 && col == 0) {
                    partial_sum_acc[item.get_group_linear_id()] = local_cache[0][0];
                }
            });
        });
        sum_event.wait();

        // sum up the partial sums on the host device
        auto partial_sums_acc = partial_sums.get_host_access();
        auto sum = std::reduce(std::execution::par, partial_sums_acc.begin(), partial_sums_acc.end());

        return full ? std::sqrt(sum) : sum;
    }

  private:
    ::sycl::queue &queue_;
    const matrix_view<matrix_type::general> &A_;
};

}  // namespace plssvm::sycl::linalg
#endif  // PLSSVM_BACKENDS_SYCL_LINALG_NORMS_HPP_
