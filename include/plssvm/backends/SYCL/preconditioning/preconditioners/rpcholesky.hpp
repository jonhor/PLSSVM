#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"
#include "plssvm/backends/SYCL/preconditioning/sycl_preconditioner.hpp"

namespace plssvm::sycl::preconditioning {

using linalg::matrix, linalg::matrix_type;

class rpcholesky_preconditioner : public sycl_preconditioner {
    rpcholesky_preconditioner(::sycl::queue &queue, matrix<matrix_type::general> &&M) :
        sycl_preconditioner(queue),
        M_(std::move(M)) {
    }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) override {
        linalg::matrix_multiplication(queue_, M_, B, C);
        linalg::matrix_addition(queue_, C, real_type{ 1 } / cost_factor_, B);
    }

    // TODO implement custom product

    const double cost_factor_ = real_type{ 1 };
    matrix<matrix_type::general> M_;

    friend class rpcholesky_preconditioner_constructor;
};

class rpcholesky_preconditioner_constructor {
  public:
    rpcholesky_preconditioner_constructor(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K, real_type c) :
        queue_(queue),
        K_(K),
        c_(c) { }

    rpcholesky_preconditioner operator()() {
        auto G = linalg::randomly_pivoted_cholesky{ queue_, K_, 300 }();
        auto F = linalg::transposed(queue_, G);
        auto [U, S] = linalg::svd(queue_, F);
        transform_sigma(S);

        auto UT = linalg::transposed(queue_, U);
        auto V = linalg::matrix_multiplication(queue_, U.view(), S.view());
        auto M = linalg::matrix_multiplication(queue_, V.view(), UT.view());

        return rpcholesky_preconditioner{ queue_, std::move(M) };
    }

  private:
    void transform_sigma(matrix_view<matrix_type::diagonal> &S) {
        const auto N = std::min(S.n_rows, S.n_cols);
        ::sycl::nd_range nd_range{ ::sycl::range(N), ::sycl::range(linalg::BLOCK_SIZE * linalg::BLOCK_SIZE) };

        const auto c = c_;
        auto event = queue_.parallel_for<class transform_sigma>(nd_range, [=](const ::sycl::nd_item<1> &item) {
            const auto global_idx = item.get_global_id();

            if (global_idx >= N) {
                return;
            }

            S(global_idx, global_idx) = (real_type{ 1 } / (S(global_idx, global_idx) * S(global_idx, global_idx) + c)) - (real_type{ 1 } / c);
        });
        event.wait();
    }
    
    ::sycl::queue queue_;
    matrix_view<matrix_type::symmetric> K_;
    const real_type c_;
};

}  // namespace plssvm::sycl::preconditioning

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_RPCHOLESKY_HPP_
