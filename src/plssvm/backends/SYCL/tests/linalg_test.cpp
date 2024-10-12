#include "plssvm/backends/SYCL/detail/linalg.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace plssvm;
using namespace plssvm::sycl::detail;

#define abs_err 1e-6

/*
 * *******************************
 * * Triangular Solve (TRSM)
 * *******************************
 */
TEST(LinearAlgebra, TriangularSolve_Lower1) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::lower>({ 1, 2, 3, 4, 5, 6 }, 3, 3, queue);
    auto B = helper::create_shared_view<matrix_type::general>({ 1, 2, 3, 4, 5, 6 }, 3, 2, queue);
    auto X = helper::zeros_like(B, queue);

    linalg::blas::trsm(A, B, X, queue);

    EXPECT_NEAR(X(0, 0), 1.0, abs_err);
    EXPECT_NEAR(X(0, 1), 2.0, abs_err);
    EXPECT_NEAR(X(1, 0), 1.0 / 3.0, abs_err);
    EXPECT_NEAR(X(1, 1), 0.0, abs_err);
    EXPECT_NEAR(X(2, 0), -1.0 / 9.0, abs_err);
    EXPECT_NEAR(X(2, 1), -1.0 / 3.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(B.data(), queue);
    ::sycl::free(X.data(), queue);
}

TEST(LinearAlgebra, TriangularSolve_Lower2) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::lower>({ 3, 2, 1, 1, 0, 1, 1, 1, 1, 1 }, 4, 4, queue);
    auto B = helper::create_shared_view<matrix_type::general>({ 4, 2, 4, 2 }, 4, 1, queue);
    auto X = helper::zeros_like(B, queue);

    linalg::blas::trsm(A, B, X, queue);

    EXPECT_NEAR(X(0, 0), 4.0 / 3.0, abs_err);
    EXPECT_NEAR(X(1, 0), -2.0 / 3.0, abs_err);
    EXPECT_NEAR(X(2, 0), 8.0 / 3.0, abs_err);
    EXPECT_NEAR(X(3, 0), -4.0 / 3.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(B.data(), queue);
    ::sycl::free(X.data(), queue);
}

TEST(LinearAlgebra, TriangularSolve_Lower3) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::lower>({ 2, 6, 1, -8, 5, 3 }, 3, 3, queue);
    auto B = helper::create_shared_view<matrix_type::general>({ 1, 2, 3 }, 3, 1, queue);
    auto X = helper::zeros_like(B, queue);

    linalg::blas::trsm(A, B, X, queue);

    EXPECT_NEAR(X(0, 0), 1.0 / 2.0, abs_err);
    EXPECT_NEAR(X(1, 0), -1.0, abs_err);
    EXPECT_NEAR(X(2, 0), 4.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(B.data(), queue);
    ::sycl::free(X.data(), queue);
}

TEST(LinearAlgebra, TriangularSolve_Upper1) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 6, 2, 3, 8, 9 }, 3, 3, queue);
    auto B = helper::create_shared_view<matrix_type::general>({ 1, 2, 3, 4, 5, 6 }, 3, 2, queue);
    auto X = helper::zeros_like(B, queue);

    linalg::blas::trsm(A, B, X, queue);

    EXPECT_NEAR(X(0, 0), 25.0 / 36.0, abs_err);
    EXPECT_NEAR(X(0, 1), 5.0 / 6.0, abs_err);
    EXPECT_NEAR(X(1, 0), -13.0 / 27.0, abs_err);
    EXPECT_NEAR(X(1, 1), -4.0 / 9.0, abs_err);
    EXPECT_NEAR(X(2, 0), 5.0 / 9.0, abs_err);
    EXPECT_NEAR(X(2, 1), 6.0 / 9.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(B.data(), queue);
    ::sycl::free(X.data(), queue);
}

/*
 * *******************************
 * * (Direct) Cholesky Decomposition
 * *******************************
 */
TEST(LinearAlgebra, DirectCholesky) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 12, -16, 37, -43, 98 }, 3, 3, queue);
    auto U = helper::zeros_like(A, queue);

    linalg::direct_cholesky(A, U, queue);
    EXPECT_NEAR(U(0, 0), 2.0, abs_err);
    EXPECT_NEAR(U(0, 1), 6.0, abs_err);
    EXPECT_NEAR(U(0, 2), -8.0, abs_err);
    EXPECT_NEAR(U(1, 1), 1.0, abs_err);
    EXPECT_NEAR(U(1, 2), 5.0, abs_err);
    EXPECT_NEAR(U(2, 2), 3.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(U.data(), queue);
}
