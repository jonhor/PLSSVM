#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_DUMMY_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_DUMMY_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/backends/SYCL/preconditioning/sycl_preconditioner.hpp"

#include "sycl/sycl.hpp"

namespace plssvm::sycl::preconditioning {

using linalg::matrix_view, linalg::matrix, linalg::matrix_type;

class dummy_preconditioner : public sycl_preconditioner {
    dummy_preconditioner(::sycl::queue &queue, matrix<matrix_type::general> &&M) :
        sycl_preconditioner(queue),
        M_(std::move(M)) {
    }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) override {
        linalg::matrix_multiplication<matrix_type::general>(queue_, M_, B, C);
    }

    matrix<matrix_type::general> M_;

    friend class dummy_preconditioner_constructor;
};

class dummy_preconditioner_constructor {
  public:
    dummy_preconditioner_constructor(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K) :
        queue_(queue),
        K_(K) { }

    dummy_preconditioner operator()() {
        auto I = linalg::identity(queue_, K_.n_rows, K_.n_cols, K_.padding);
        return dummy_preconditioner{ queue_, std::move(I) };
    }

  private:
    ::sycl::queue queue_;
    matrix_view<matrix_type::symmetric> K_;
};

}  // namespace plssvm::sycl::preconditioning

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_DUMMY_HPP_
