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
TEST(Preconditioning, Cholesky_Linalg) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 12, -16, 37, -43, 98 }, 3, 3, queue);
    auto r = helper::create_shared_view<matrix_type::general>({ 1, 2, 3 }, 3, 1, queue);

    auto U = helper::zeros_like(A, queue);
    linalg::direct_cholesky(A, U, queue);

    EXPECT_NEAR(U(0, 0), 2.0, abs_err);
    EXPECT_NEAR(U(0, 1), 6.0, abs_err);
    EXPECT_NEAR(U(0, 2), -8.0, abs_err);
    EXPECT_NEAR(U(1, 1), 1.0, abs_err);
    EXPECT_NEAR(U(1, 2), 5.0, abs_err);
    EXPECT_NEAR(U(2, 2), 3.0, abs_err);

    // apply preconditioner
    auto UT = helper::transpose(U, queue);
    auto y = helper::zeros_like(r, queue);
    linalg::blas::trsm(UT, r, y, queue);

    EXPECT_NEAR(y(0, 0), 1.0 / 2.0, abs_err);
    EXPECT_NEAR(y(1, 0), -1.0, abs_err);
    EXPECT_NEAR(y(2, 0), 4.0, abs_err);

    auto z = helper::zeros_like(r, queue);
    linalg::blas::trsm(U, y, z, queue);

    EXPECT_NEAR(z(0, 0), 28.0 + 7.0 / 12.0, abs_err);
    EXPECT_NEAR(z(1, 0), -(7.0 + 2.0 / 3.0), abs_err);
    EXPECT_NEAR(z(2, 0), 4.0 / 3.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(r.data(), queue);
    ::sycl::free(y.data(), queue);
    ::sycl::free(z.data(), queue);
    ::sycl::free(U.data(), queue);
    ::sycl::free(UT.data(), queue);
}

TEST(Preconditioning, Cholesky) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 12, -16, 37, -43, 98 }, 3, 3, queue);
    auto M = helper::zeros_like(A, queue);

    std::vector<real_type> R_vec{ 1, 2, 3 };
    soa_matrix<real_type> R(shape(3, 1), R_vec.data());
    soa_matrix<real_type> S(shape(3, 1));

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

// TODO test the cholesky when everything is padded, e.g. soa_matrix and matrix_view etc.
TEST(Preconditioning, CholeskyPadded) {
    EXPECT_EQ(true, true);
}
