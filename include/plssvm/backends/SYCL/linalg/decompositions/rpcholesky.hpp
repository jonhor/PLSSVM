#ifndef PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_

#include "plssvm/backends/SYCL/detail/random.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"

#include "sycl/sycl.hpp"

#include <optional>

namespace plssvm::sycl::linalg {

/**
 * Implements the RPCholesky algorithm from the paper
 * Robust, randomized preconditioning for kernel ridge regression.
 */
class randomly_pivoted_cholesky {
  public:
    randomly_pivoted_cholesky(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K, unsigned int k = 0) :
        queue_(queue),
        nd_range_(::sycl::range<1>(K.n_rows), ::sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE)),
        N_(K.n_rows),
        probabilities_{ ::sycl::range<1>(N_), ::sycl::property::no_init() },
        K_(K),
        D_(linalg::diagonal(queue, K)),
        D_host(N_) {
        if (k == 0) {
            k = std::max(static_cast<unsigned int>(std::sqrt(N_)), 300u);
            k = std::min(k, static_cast<unsigned int>(static_cast<double>(N_) / 2.0));  // dont oversample
        }

        k_ = k;
    }

    randomly_pivoted_cholesky(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &A, const matrix_view<matrix_type::general> &pivots, unsigned int k = 0) :
        randomly_pivoted_cholesky(queue, A, k) {
        PLSSVM_ASSERT(pivots.n_rows == 1, "pivots is expected to be one-dimensional");
        PLSSVM_ASSERT(k == pivots.n_cols, "k should be equal to the number of pivots");

        for (std::size_t i = 0; i < pivots.n_cols; ++i) {
            pivots_.push_back(static_cast<std::size_t>(pivots(0, i)));
        }
    }

    inline matrix<matrix_type::general> operator()() {
        // k x N matrix
        auto G = linalg::zeros<matrix_type::general>(queue_, k_, N_, PADDING_SIZE);
        auto G_view = G.view();

        // call kernel in loop, updating probabilities on the host in between.
        detail::rng rng{};

        const bool use_pivots = pivots_.size() > 0;
        for (std::size_t i = 0; i < k_; ++i) {
            std::size_t row_idx;
            if (use_pivots) {
                row_idx = pivots_[i];
            } else {
                update_probabilities();

                // Optimization:
                // We currently use the C++ stdlib random interface to choose the next row.
                // In a perfect world this is an operation that can be performed on the computation device.
                auto probabilities = probabilities_.get_host_access();
                row_idx = rng.choice<std::size_t>(probabilities.begin(), probabilities.end());
            }

            update_approximation(G_view, i, row_idx);
        }
        return G;
    }

  private:
    inline void update_probabilities() {
        auto D = D_.view();

        // Optimization:
        // This is a reduction operation that can be performed with the SYCL reduction interface.
        // AdaptiveCpp does not fully support it at the current time.
        // A showcase on how to implement a reduction operation from scratch can be seen in linalg/norms.hpp (frobenius).
        // Because of time constraints we copy the data to a host memory location and reduce it with the stdlib interfaces provided by C++.
        queue_.copy<real_type>(D.data(), D_host.data(), N_);
        const auto diag_sum = std::reduce(D_host.begin(), D_host.end());

        ::sycl::nd_range<1> nd_range{ ::sycl::range<1>(N_), ::sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE) };
        auto event = queue_.submit([&](::sycl::handler &cgh) {
            auto probabilities = probabilities_.get_access<::sycl::access::mode::discard_write>(cgh);
            cgh.parallel_for<class rpcholesky_update_probabilities>(nd_range, [=](::sycl::nd_item<1> item) {
                const auto global_id = item.get_global_id();
                probabilities[global_id] = D(global_id, global_id) / diag_sum;
            });
        });
        event.wait();
    }

    inline void update_approximation(matrix_view<matrix_type::general> &G, std::size_t i, std::size_t row_idx) {
        // fmt::println("{} iteration", i);

        auto D = D_.view();
        // TODO check if d is zero
        auto event = queue_.submit([&](::sycl::handler &cgh) {
            const auto N = N_;
            const auto K = K_;

            cgh.parallel_for<class rpcholesky_update_approximation>(nd_range_, [=](::sycl::nd_item<1> item) {
                // const auto global_id = item.get_global_id();
                const auto global_id = item.get_global_id(0);
                const auto d = D(row_idx, row_idx);

                if (global_id >= N) {
                    return;
                }

                // load relevant row r = K[idx, :]
                real_type r = K(row_idx, global_id);

                // G[:i, idx].T @ G[:i,:]
                auto dot = real_type{ 0 };
                for (std::size_t j = 0; j < i; ++j) {
                    dot += G(j, row_idx) * G(j, global_id);
                }
                r -= dot;

                const auto g = r / std::sqrt(d);
                D(global_id, global_id) = std::max(real_type{ 0 }, D(global_id, global_id) - g * g);
                G(i, global_id) = g;
            });
        });
        event.wait();
    }

    ::sycl::queue queue_;
    ::sycl::nd_range<1> nd_range_;

    std::vector<std::size_t> pivots_{};

    const std::size_t N_;
    ::sycl::buffer<real_type, 1> probabilities_;  // probabilities that a specific row from the kernel matrix is chosen to update our approximation

    const matrix_view<matrix_type::symmetric> &K_;  // kernel matrix
    const matrix<matrix_type::diagonal> D_;         // diagonal of the kernel matrix
    std::vector<real_type> D_host;                  // this provides a host memory location for calculating the sum of diagonal elements (can be optimized away).

    unsigned int k_;
};
}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_
