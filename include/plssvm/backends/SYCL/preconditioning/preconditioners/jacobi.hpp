#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_JACOBI_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_JACOBI_HPP_
#pragma once

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/backends/SYCL/preconditioning/sycl_preconditioner.hpp"

namespace plssvm::sycl::preconditioning {

using linalg::matrix_view, linalg::matrix, linalg::matrix_type;

class jacobi_preconditioner : public sycl_preconditioner {
    jacobi_preconditioner(::sycl::queue &queue, matrix<matrix_type::diagonal> &&M) :
        sycl_preconditioner(queue),
        M_(std::move(M)) {
    }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) override {
        linalg::matrix_multiplication(queue_, M_, B, C);
    }

    matrix<matrix_type::diagonal> M_;

    friend class jacobi_preconditioner_constructor;
};

class jacobi_preconditioner_constructor {
  public:
    jacobi_preconditioner_constructor(::sycl::queue &queue, const matrix_view<matrix_type::symmetric> &K) :
        queue_(queue),
        K_(K) { }

    jacobi_preconditioner operator()() {
        auto D = linalg::diagonal(queue_, K_);
        linalg::invert(queue_, D);
        return jacobi_preconditioner{ queue_, std::move(D) };
    }

  private:
    ::sycl::queue queue_;
    matrix_view<matrix_type::symmetric> K_;
};

}  // namespace plssvm::sycl::preconditioning

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_PRECONDITIONERS_JACOBI_HPP_
