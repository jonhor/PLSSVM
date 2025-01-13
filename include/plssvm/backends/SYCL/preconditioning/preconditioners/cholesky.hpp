#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_CHOLESKY_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_CHOLESKY_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/backends/SYCL/preconditioning/sycl_preconditioner.hpp"
#include "plssvm/detail/logging.hpp"

#include "sycl/sycl.hpp"

namespace plssvm::sycl::preconditioning {

using linalg::matrix_view, linalg::matrix, linalg::matrix_type;

class cholesky_preconditioner : public sycl_preconditioner {
    cholesky_preconditioner(::sycl::queue &queue, matrix<matrix_type::upper> &&M) :
        sycl_preconditioner(queue),
        M_(std::move(M)),
        MT_(linalg::transposed(queue, M_)) { }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) override {
        linalg::triangular_solve_lower_gpu(queue_, MT_.view(), B);
        linalg::triangular_solve_upper_gpu(queue_, M_.view(), B);
        queue_.memcpy(C.data(), B.data(), C.size_bytes_padded()).wait();
    }

    matrix<matrix_type::upper> M_;
    /*
     * Optimize:
     * Implement a transpose method on matrix_view that only changes the indexing of the view instead of copying data (similar to numpy).
     * So we don't have to store the same values twice.
     */
    matrix<matrix_type::lower> MT_;

    friend class cholesky_preconditioner_constructor;
};

class cholesky_preconditioner_constructor {
  public:
    cholesky_preconditioner_constructor(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K) :
        queue_(queue),
        K_(K) { }

    cholesky_preconditioner operator()() {
        auto cholesky = linalg::cholesky_decomposition{ queue_, K_ };
        auto U = cholesky();

        plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                            "\nCholesky decomposition timings\nblock factorization: {}.\nblock solve: {}.\nblock update: {}.\n",
                            cholesky.total_factorization_time(),
                            cholesky.total_solve_time(),
                            cholesky.total_update_time());
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((::plssvm::detail::tracking::tracking_entry{ "preconditioner", "factorization_time", cholesky.total_factorization_time() }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((::plssvm::detail::tracking::tracking_entry{ "preconditioner", "solve_time", cholesky.total_solve_time() }));
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((::plssvm::detail::tracking::tracking_entry{ "preconditioner", "update_time", cholesky.total_update_time() }));

        return cholesky_preconditioner{ queue_, std::move(U) };
    }

  private:
    ::sycl::queue queue_;
    matrix_view<matrix_type::symmetric> K_;
};

}  // namespace plssvm::sycl::preconditioning

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_CHOLESKY_HPP_
