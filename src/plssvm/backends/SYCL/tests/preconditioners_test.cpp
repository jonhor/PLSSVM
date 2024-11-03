#include "plssvm/backends/SYCL/detail/linalg.hpp"
#include "plssvm/backends/SYCL/detail/preconditioners.hpp"
#include "plssvm/matrix.hpp"
#include "plssvm/shape.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace plssvm;
using namespace plssvm::sycl::detail;

#define abs_err 1e-6

/*
 * *******************************
 * * Cholesky Preconditioner
 * *******************************
 */
TEST(Preconditioners, Cholesky_Linalg) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 4, 12, -16, 37, -43, 98 }, 3, 3);
    auto r = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3 }, 3, 1);

    auto U = utility::zeros<matrix_type::upper>(queue, A->shape);
    linalg::direct_cholesky(A, U, queue);

    EXPECT_NEAR(U(0, 0), 2.0, abs_err);
    EXPECT_NEAR(U(0, 1), 6.0, abs_err);
    EXPECT_NEAR(U(0, 2), -8.0, abs_err);
    EXPECT_NEAR(U(1, 1), 1.0, abs_err);
    EXPECT_NEAR(U(1, 2), 5.0, abs_err);
    EXPECT_NEAR(U(2, 2), 3.0, abs_err);

    // apply preconditioner
    auto UT = utility::transpose(queue, U);
    auto y = utility::zeros<matrix_type::general>(queue, r->shape);
    linalg::blas::trsm(queue, UT.view(), r.view(), y.view());

    EXPECT_NEAR(y(0, 0), 1.0 / 2.0, abs_err);
    EXPECT_NEAR(y(1, 0), -1.0, abs_err);
    EXPECT_NEAR(y(2, 0), 4.0, abs_err);

    auto z = utility::zeros<matrix_type::general>(queue, r->shape);
    linalg::blas::trsm(queue, U.view(), y, z);

    EXPECT_NEAR(z(0, 0), 28.0 + 7.0 / 12.0, abs_err);
    EXPECT_NEAR(z(1, 0), -(7.0 + 2.0 / 3.0), abs_err);
    EXPECT_NEAR(z(2, 0), 4.0 / 3.0, abs_err);
}

TEST(Preconditioners, Cholesky) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 4, 12, -16, 37, -43, 98 }, 3, 3);
    auto M = utility::zeros<matrix_type::upper>(queue, A->shape);

    std::vector<real_type> R_vec{ 1, 2, 3 };
    // shapes of R and S are transposed in conjugate_gradients()
    soa_matrix<real_type> R(shape(3, 1), R_vec.data());
    soa_matrix<real_type> S(shape(3, 1));

    auto apply_preconditioner = precond::cholesky(queue, A, M);

    EXPECT_NEAR(M(0, 0), 2.0, abs_err);
    EXPECT_NEAR(M(0, 1), 6.0, abs_err);
    EXPECT_NEAR(M(0, 2), -8.0, abs_err);
    EXPECT_NEAR(M(1, 1), 1.0, abs_err);
    EXPECT_NEAR(M(1, 2), 5.0, abs_err);
    EXPECT_NEAR(M(2, 2), 3.0, abs_err);

    apply_preconditioner(R, S);

    EXPECT_NEAR(S(0, 0), 28.0 + 7.0 / 12.0, abs_err);
    EXPECT_NEAR(S(1, 0), -(7.0 + 2.0 / 3.0), abs_err);
    EXPECT_NEAR(S(2, 0), 4.0 / 3.0, abs_err);
}

/*
TEST(Preconditioning, CholeskyPadded) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 12, -16, 0, 0, 37, -43, 0, 0, 98, 0, 0, 0, 0, 0 }, 3, 3, 2, queue);
    auto M = helper::zeros_like(A, queue);

    std::vector<real_type> R_vec{ 1, 2, 3 };
    // shapes of R and S are transposed in conjugate_gradients()
    soa_matrix<real_type> R(shape(1, 3), R_vec.data());
    soa_matrix<real_type> S(shape(1, 3));

    auto apply_preconditioner = precond::cholesky(A, M, queue);

    EXPECT_NEAR(M(0, 0), 2.0, abs_err);
    EXPECT_NEAR(M(0, 1), 6.0, abs_err);
    EXPECT_NEAR(M(0, 2), -8.0, abs_err);
    EXPECT_NEAR(M(1, 1), 1.0, abs_err);
    EXPECT_NEAR(M(1, 2), 5.0, abs_err);
    EXPECT_NEAR(M(2, 2), 3.0, abs_err);

    apply_preconditioner(R, S);

    EXPECT_NEAR(S(0, 0), 28.0 + 7.0 / 12.0, abs_err);
    EXPECT_NEAR(S(1, 0), -(7.0 + 2.0 / 3.0), abs_err);
    EXPECT_NEAR(S(2, 0), 4.0 / 3.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(M.data(), queue);
}
*/
