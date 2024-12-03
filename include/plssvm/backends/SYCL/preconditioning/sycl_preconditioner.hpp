#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/preconditioner.hpp"

namespace plssvm::sycl {

using linalg::matrix_type, linalg::matrix_view;

class sycl_preconditioner : public preconditioner {
  public:
    virtual void apply(const soa_matrix<real_type> &B, soa_matrix<real_type> &C) {
        // rows and columns are switched
        // TODO this code should be moved to the constructor? e.g. can we only allocate memory once and reuse it?

        auto B_ = linalg::empty<matrix_type::general>(queue_, B.num_cols(), B.num_rows(), B.padding().x);
        auto C_ = linalg::empty<matrix_type::general>(queue_, C.num_cols(), C.num_rows(), C.padding().x);

        // auto B_ = linalg::empty<matrix_type::general>(queue_, B.num_rows(), B.num_cols(), B.padding().x);
        // auto C_ = linalg::empty<matrix_type::general>(queue_, C.num_rows(), C.num_cols(), C.padding().x);

        auto B_view = B_.view();
        auto C_view = C_.view();

        copy_from_host_matrix(queue_, B_view, B);
        copy_from_host_matrix(queue_, C_view, C);

        this->apply(B_view, C_view);

        copy_to_host_matrix(queue_, C_view, C);
    }

  protected:
    sycl_preconditioner(::sycl::queue &queue) :
        queue_(queue) { }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) = 0;

    ::sycl::queue queue_;
};

}  // namespace plssvm::sycl

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_
