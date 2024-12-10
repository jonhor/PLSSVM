#ifndef PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_
#define PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/preconditioner.hpp"

namespace plssvm::sycl {

using linalg::matrix_type, linalg::matrix_view;

class sycl_preconditioner : public preconditioner {
  public:
    virtual std::chrono::duration<long, std::milli> apply(const soa_matrix<real_type> &B, soa_matrix<real_type> &C) override {
        auto start_time = std::chrono::steady_clock::now();

        // rows and columns are switched
        // TODO this code should be moved to the constructor? e.g. can we only allocate memory once and reuse it?

        auto B_ = linalg::empty<matrix_type::general>(queue_, B.num_cols(), B.num_rows(), B.padding().x);
        auto C_ = linalg::empty<matrix_type::general>(queue_, C.num_cols(), C.num_rows(), C.padding().x);

        // auto B_ = linalg::empty<matrix_type::general>(queue_, B.num_rows(), B.num_cols(), B.padding().x);
        // auto C_ = linalg::empty<matrix_type::general>(queue_, C.num_rows(), C.num_cols(), C.padding().x);

        auto B_view = B_.view();
        auto C_view = C_.view();

        copy_from_host_matrix(queue_, B_view, B);
        // TODO this can be removed p sure
        copy_from_host_matrix(queue_, C_view, C);

        this->apply(B_view, C_view);

        copy_to_host_matrix(queue_, C_view, C);

        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    }

    virtual std::chrono::duration<long, std::milli> custom_product(const soa_matrix<real_type> &D, soa_matrix<real_type> &Q) override {
        auto start_time = std::chrono::steady_clock::now();

        auto D_ = linalg::empty<matrix_type::general>(queue_, D.num_cols(), D.num_rows(), D.padding().x);
        auto Q_ = linalg::empty<matrix_type::general>(queue_, Q.num_cols(), Q.num_rows(), Q.padding().x);

        auto D_view = D_.view();
        auto Q_view = Q_.view();

        copy_from_host_matrix(queue_, D_view, D);
        copy_from_host_matrix(queue_, Q_view, Q);

        this->custom_product(D_view, Q_view);

        copy_to_host_matrix(queue_, Q_view, Q);

        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    }

  protected:
    sycl_preconditioner(::sycl::queue &queue) :
        queue_(queue) { }

    virtual void apply(matrix_view<matrix_type::general> &B, matrix_view<matrix_type::general> &C) = 0;

    [[maybe_unused]] virtual void custom_product(matrix_view<matrix_type::general> &D, matrix_view<matrix_type::general> &Q) { }

    ::sycl::queue queue_;
};

}  // namespace plssvm::sycl

#endif  // PLSSVM_BACKENDS_SYCL_PRECONDITIONING_SYCL_PRECONDITIONER_HPP_
