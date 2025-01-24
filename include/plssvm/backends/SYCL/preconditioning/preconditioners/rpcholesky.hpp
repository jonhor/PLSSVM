#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/utility.hpp"
#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"
#include "plssvm/backends/SYCL/preconditioning/sycl_preconditioner.hpp"
#include "plssvm/detail/tracking/performance_tracker.hpp"

#ifndef RUNNING_GTEST
    #include "plssvm/detail/logging.hpp"
#endif

namespace plssvm::sycl::preconditioning {

using linalg::matrix, linalg::matrix_type;

namespace internal {
void transform_sigma(::sycl::queue &queue, matrix_view<matrix_type::diagonal> &S, real_type c) {
    const auto N = std::min(S.n_rows, S.n_cols);

    auto nd_range = detail::get_uniform_1d_range(N, linalg::MAX_WORKGROUP_SIZE);
    auto event = queue.parallel_for<class transform_sigma>(nd_range, [=](const ::sycl::nd_item<1> &item) {
        const auto global_idx = item.get_global_id();

        if (global_idx >= N) {
            return;
        }

        S(global_idx, global_idx) = (real_type{ 1 } / (S(global_idx, global_idx) * S(global_idx, global_idx) + c)) - (real_type{ 1 } / c);
    });
    event.wait();
}
}  // namespace internal

class rpcholesky_preconditioner : public sycl_preconditioner {
    rpcholesky_preconditioner(::sycl::queue &queue, matrix_view<matrix_type::symmetric> &K, matrix<matrix_type::general> &&U, matrix<matrix_type::diagonal> &&S, real_type c) :
        sycl_preconditioner(queue),
        K_(K),
        U_(std::move(U)),
        S_(std::move(S)),
        c_(c) {
    }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) override {
        auto UT = linalg::transposed(queue_, U_);
        linalg::matrix_multiplication(queue_, UT.view(), B, C);
        auto V = linalg::matrix_multiplication(queue_, S_.view(), C);
        linalg::matrix_multiplication(queue_, U_.view(), V.view(), C);
        linalg::matrix_addition(queue_, C, real_type{ 1 } / c_, B);
    }

    virtual void custom_product(matrix_view<matrix_type::general> &D, matrix_view<matrix_type::general> &Q) override {
        linalg::matrix_multiplication(queue_, K_, D, Q);
        linalg::matrix_addition(queue_, Q, c_, D);
    }

    virtual bool has_custom_product() override {
        return true;
    }

    virtual bool recalculate_residuals() override {
        return false;
    }

    matrix_view<matrix_type::symmetric> K_;
    matrix<matrix_type::general> U_;
    matrix<matrix_type::diagonal> S_;
    const real_type c_;

    friend class rpcholesky_preconditioner_constructor;
};

class rpcholesky_preconditioner_constructor {
  public:
    rpcholesky_preconditioner_constructor(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K, real_type c) :
        queue_(queue),
        K_(K),
        c_(c) { }

    rpcholesky_preconditioner operator()() {
        std::chrono::steady_clock::time_point start_time, end_time;

        // Calculate low-rank kernel approximation
        start_time = std::chrono::steady_clock::now();
        auto G = linalg::randomly_pivoted_cholesky{ queue_, K_ }();
        end_time = std::chrono::steady_clock::now();
        auto rpcholesky_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        auto F = linalg::transposed(queue_, G);

        // Compute thin SVD
        start_time = std::chrono::steady_clock::now();
        auto [U, S] = linalg::svd(queue_, F);
        end_time = std::chrono::steady_clock::now();
        auto svd_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        internal::transform_sigma(queue_, S, c_);

        plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                            "\nRandomly Pivoted Cholesky timings\nCompute RPCholesky approximation: {}.\nSVD: {}.\n",
                            rpcholesky_time,
                            svd_time);
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((::plssvm::detail::tracking::tracking_entry{ "preconditioner", "compute_approximation", rpcholesky_time }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((::plssvm::detail::tracking::tracking_entry{ "preconditioner", "svd_compute_time", svd_time }));

        return rpcholesky_preconditioner{ queue_, K_, std::move(U), std::move(S), c_ };
    }

  private:
    ::sycl::queue queue_;
    matrix_view<matrix_type::symmetric> K_;
    const real_type c_;
};

}  // namespace plssvm::sycl::preconditioning

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
