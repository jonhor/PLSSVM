#ifndef PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_

#include "plssvm/backends/SYCL/detail/random.hpp"
#include "plssvm/backends/SYCL/detail/utility.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/detail/assert.hpp"

#include "sycl/sycl.hpp"

#include <execution>
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
        N_(K.n_rows),
        K_(K),
        D_(linalg::diagonal(queue, K)) {
        if (k == 0) {
            k = std::max(static_cast<unsigned int>(std::sqrt(N_)), 300u);
            k = std::min(k, static_cast<unsigned int>(static_cast<double>(N_) / 2.0));  // don't oversample
        }

        k_ = k;

        probabilities_ = ::sycl::malloc_host<real_type>(N_, queue_);
        queue_.fill<real_type>(probabilities_, 1, N_).wait();
    }

    randomly_pivoted_cholesky(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &A, const matrix_view<matrix_type::general> &pivots, unsigned int k = 0) :
        randomly_pivoted_cholesky(queue, A, k) {
        PLSSVM_ASSERT(pivots.n_rows == 1, "pivots is expected to be one-dimensional");
        PLSSVM_ASSERT(k == pivots.n_cols, "k should be equal to the number of pivots");

        for (std::size_t i = 0; i < pivots.n_cols; ++i) {
            pivots_.push_back(static_cast<std::size_t>(pivots(0, i)));
        }
    }

    ~randomly_pivoted_cholesky() {
        ::sycl::free(probabilities_, queue_);
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
                row_idx = rng.choice<std::size_t>(probabilities_, probabilities_ + N_);
            }

            update_approximation(G_view, i, row_idx);
        }
        return G;
    }

  private:
    inline void update_probabilities() {
        auto D = D_.view();

        queue_.copy<real_type>(D.data(), probabilities_, N_).wait();
        const auto diag_sum = std::reduce(std::execution::par, probabilities_, probabilities_ + N_);

        auto probabilities = probabilities_;
        auto nd_range = detail::get_uniform_1d_range(N_, MAX_WORKGROUP_SIZE);
        auto event = queue_.submit([&](::sycl::handler &cgh) {
            auto N = N_;

            cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> item) {
                const auto global_id = item.get_global_id(0);

                if (global_id < N) {
                    probabilities[global_id] /= diag_sum;
                }
            });
        });
        event.wait();
    }

    inline void update_approximation(matrix_view<matrix_type::general> &G, std::size_t i, std::size_t row_idx) {
        auto D = D_.view();

        auto nd_range = detail::get_uniform_1d_range(N_, MAX_WORKGROUP_SIZE);
        auto smallest_eps = std::numeric_limits<real_type>::epsilon();

        auto event = queue_.submit([&](::sycl::handler &cgh) {
            const auto N = N_;
            const auto K = K_;

            auto d = std::max(D(row_idx, row_idx), smallest_eps);
            auto d_sqrt = std::sqrt(d);
            cgh.parallel_for(nd_range, [=](::sycl::nd_item<1> item) {
                const auto global_id = item.get_global_id(0);

                if (global_id < N) {
                    // load relevant row r = K[idx, :]
                    auto r = K(row_idx, global_id);

                    // G[:i, idx].T @ G[:i,:]
                    auto dot = real_type{ 0 };
                    for (std::size_t j = 0; j < i; ++j) {
                        dot += G(j, row_idx) * G(j, global_id);
                    }
                    r -= dot;

                    auto g = r / d_sqrt;
                    D(global_id, global_id) = std::max(real_type{ 0 }, D(global_id, global_id) - g * g);
                    G(i, global_id) = g;
                }
            });
        });
        event.wait();
    }

    ::sycl::queue queue_;

    std::vector<std::size_t> pivots_{};

    const std::size_t N_;

    const matrix_view<matrix_type::symmetric> &K_;  // kernel matrix
    const matrix<matrix_type::diagonal> D_;         // diagonal of the kernel matrix
    real_type *probabilities_;

    unsigned int k_;
};

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_RPCHOLESKY_HPP_
