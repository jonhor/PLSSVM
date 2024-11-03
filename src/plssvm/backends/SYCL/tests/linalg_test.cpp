#include "plssvm/backends/SYCL/detail/block.hpp"
#include "plssvm/backends/SYCL/detail/linalg.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace plssvm;
using namespace plssvm::sycl::detail;

#define abs_err 1e-6

/*
 * *******************************
 * * General Matrix Multiplication (GEMM)
 * *******************************
 */
TEST(LinearAlgebra, GEMM_Symmetric) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 64;

    auto A = utility::randn<matrix_type::general>(queue, N, N, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, N, N, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, N, PADDING_SIZE);

    linalg::blas::gemm(queue, A.view(), B.view(), C.view());

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < N; ++j) {
            real_type sum = 0;
            for (auto k = 0; k < N; ++k) {
                sum += A(i, k) * B(k, j);
            }
            EXPECT_NEAR(C(i, j), sum, abs_err);
        }
    }
}

TEST(LinearAlgebra, GEMM_SymmetricLarge) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 1024;

    auto A = utility::randn<matrix_type::general>(queue, N, N, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, N, N, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, N, PADDING_SIZE);

    linalg::blas::gemm(queue, A.view(), B.view(), C.view());
}

TEST(LinearAlgebra, GEMM_NonSymmetric) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 64;
    const std::size_t K = 32;
    const std::size_t M = 128;

    auto A = utility::randn<matrix_type::general>(queue, N, K, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, K, M, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, M, PADDING_SIZE);

    linalg::blas::gemm(queue, A.view(), B.view(), C.view());

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < M; ++j) {
            real_type sum = 0;
            for (auto k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            EXPECT_NEAR(C(i, j), sum, abs_err);
        }
    }
}

TEST(LinearAlgebra, GEMM_StrangeDimensions) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 73;
    const std::size_t K = 87;
    const std::size_t M = 21;

    auto A = utility::randn<matrix_type::general>(queue, N, K, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, K, M, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, M, PADDING_SIZE);

    linalg::blas::gemm(queue, A.view(), B.view(), C.view());

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < M; ++j) {
            real_type sum = 0;
            for (auto k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            EXPECT_NEAR(C(i, j), sum, abs_err);
        }
    }
}

TEST(LinearAlgebra, GEMM_UnbalancedDimensions) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 200;
    const std::size_t K = 10;
    const std::size_t M = 50;

    auto A = utility::randn<matrix_type::general>(queue, N, K, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, K, M, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, M, PADDING_SIZE);

    linalg::blas::gemm(queue, A.view(), B.view(), C.view());

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < M; ++j) {
            real_type sum = 0;
            for (auto k = 0; k < K; ++k) {
                sum += A(i, k) * B(k, j);
            }
            EXPECT_NEAR(C(i, j), sum, abs_err);
        }
    }
}

/*
 * *******************************
 * * Symmetric Matrix Multiplication (SYMM)
 * *******************************
 */
TEST(LinearAlgebra, SYMM_Basic) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 128;

    auto A = utility::randn<matrix_type::upper>(queue, N, N, PADDING_SIZE);
    auto B = utility::randn<matrix_type::general>(queue, N, N, PADDING_SIZE);
    auto C = utility::zeros<matrix_type::general>(queue, N, N, PADDING_SIZE);

    linalg::blas::symm(queue, A.view(), B.view(), C.view());

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < N; ++j) {
            real_type sum = 0;
            for (auto k = 0; k < N; ++k) {
                sum += A(i, k) * B(k, j);
            }
            EXPECT_NEAR(C(i, j), sum, abs_err);
        }
    }
}

/*
 * *******************************
 * * Triangular Solve (TRSM)
 * *******************************
 */
TEST(LinearAlgebra, TRSM_LowerSingleBlock) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::lower>(queue, { 1, 2, 3, 4, 5, 6 }, 3, 3, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 6 }, 3, 2, PADDING_SIZE);
    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);

    linalg::blas::trsm(queue, A.view(), B.view(), X.view());

    EXPECT_NEAR(X(0, 0), 1.0, abs_err);
    EXPECT_NEAR(X(0, 1), 2.0, abs_err);
    EXPECT_NEAR(X(1, 0), 1.0 / 3.0, abs_err);
    EXPECT_NEAR(X(1, 1), 0.0, abs_err);
    EXPECT_NEAR(X(2, 0), -1.0 / 9.0, abs_err);
    EXPECT_NEAR(X(2, 1), -1.0 / 3.0, abs_err);
}

TEST(LinearAlgebra, TRSM_LowerMultipleBlocks) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::lower>(queue, { 1, 5, 1, 3, 2, 2, 4, 1, 0, 4 }, 4, 4, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 4, 2, 1, 0, 2, 3, 1 }, 4, 2, PADDING_SIZE);
    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);

    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), 2);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    EXPECT_NEAR(X(0, 0), 1.0, abs_err);
    EXPECT_NEAR(X(0, 1), 4.0, abs_err);
    EXPECT_NEAR(X(1, 0), -3.0, abs_err);
    EXPECT_NEAR(X(1, 1), -19.0, abs_err);
    EXPECT_NEAR(X(2, 0), 3.0 / 2.0, abs_err);
    EXPECT_NEAR(X(2, 1), 14.0, abs_err);
    EXPECT_NEAR(X(3, 0), 1.0 / 2.0, abs_err);
    EXPECT_NEAR(X(3, 1), 1.0, abs_err);
}

TEST(LinearAlgebra, TRSM_UpperMultipleBlocks) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 5, 3, 4, 1, 2, 1, 2, 0, 4 }, 4, 4, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 4, 2, 1, 0, 2, 3, 1 }, 4, 2, PADDING_SIZE);
    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);

    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), 2);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    EXPECT_NEAR(X(0, 0), -(8.0 + 1.0 / 4.0), abs_err);
    EXPECT_NEAR(X(0, 1), 6.0 + 1.0 / 4.0, abs_err);
    EXPECT_NEAR(X(1, 0), 5.0 / 4.0, abs_err);
    EXPECT_NEAR(X(1, 1), -5.0 / 4.0, abs_err);
    EXPECT_NEAR(X(2, 0), 0.0, abs_err);
    EXPECT_NEAR(X(2, 1), 1.0, abs_err);
    EXPECT_NEAR(X(3, 0), 3.0 / 4.0, abs_err);
    EXPECT_NEAR(X(3, 1), 1.0 / 4.0, abs_err);
}

/*
 * *******************************
 * * Cholesky
 * *******************************
 */
TEST(Linalg, SolveBlockUpper) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t block_size = 3;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 5, 3, 1, 4, 1, 2, 2, 1, 2, 2, 2, 0, 0, 0 }, 5, 5, PADDING_SIZE);
    auto b = block{ block_size, 0, block_size };

    auto dummy_event = queue.single_task([=] { });

    auto event = linalg::cholesky::solve_block_upper(queue, A, b, dummy_event);
    event.wait();

    EXPECT_NEAR(A(0, 3), -2.0, abs_err);
    EXPECT_NEAR(A(1, 3), 0.0, abs_err);
    EXPECT_NEAR(A(2, 3), 1.0, abs_err);
    EXPECT_NEAR(A(0, 4), 6.0, abs_err);
    EXPECT_NEAR(A(1, 4), -1.0, abs_err);
    EXPECT_NEAR(A(2, 4), 1.0, abs_err);
}

TEST(Linalg, SolveBlock) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t block_size = 3;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 5, 3, 1, 4, 1, 2, 2, 1, 2, 2, 2, 0, 0, 0 }, 5, 5, PADDING_SIZE);
    auto b = block{ block_size, 0, block_size };

    auto dummy_event = queue.single_task([=] { });

    auto event = linalg::cholesky::solve_block(queue, A, b, dummy_event);
    event.wait();

    EXPECT_NEAR(A(0, 3), 1.0, abs_err);
    EXPECT_NEAR(A(1, 3), -3.0, abs_err);
    EXPECT_NEAR(A(2, 3), 2.5, abs_err);
    EXPECT_NEAR(A(0, 4), 4.0, abs_err);
    EXPECT_NEAR(A(1, 4), -19.0, abs_err);
    EXPECT_NEAR(A(2, 4), 14.0, abs_err);
}

TEST(Linalg, Cholesky) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 6, 8, 11, 4, 4, 3, 7, 14, 18, 7, 8, 5, 12, 26, 10, 12, 7, 17, 6, 8, 3, 8, 17, 4, 12, 3, 5, 14 }, 7, 7, PADDING_SIZE);
    auto U = utility::zeros<matrix_type::upper>(queue, A->shape, A->padding);

    fmt::println("before cholesky");

    linalg::chol(queue, A, U, 3);

    for (std::size_t i = 0; i < U->n_rows; ++i) {
        for (std::size_t j = i; j < U->n_cols; ++j) {
            fmt::print("{:.3f} ", U(i, j));
        }
        fmt::println("");
    }
}

/*
TEST(Linalg, CholeskyBasic) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 4, 12, -16, 37, -43, 98 }, 3, 3);
    auto U = utility::zeros<matrix_type::upper>(queue, A->shape);

    linalg::cholesky_rowwise(queue, A, U);

    for (std::size_t i = 0; i < U->n_rows; ++i) {
        for (std::size_t j = i; j < U->n_cols; ++j) {
            fmt::print("{} ", U(i, j));
        }
        fmt::println("");
    }

    EXPECT_NEAR(U(0, 0), 2.0, abs_err);
    EXPECT_NEAR(U(0, 1), 6.0, abs_err);
    EXPECT_NEAR(U(0, 2), -8.0, abs_err);
    EXPECT_NEAR(U(1, 1), 1.0, abs_err);
    EXPECT_NEAR(U(1, 2), 5.0, abs_err);
    EXPECT_NEAR(U(2, 2), 3.0, abs_err);
}

TEST(Linalg, CholeskyRowwisePadded) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 4, 12, -16, 37, -43, 98 }, 3, 3, PADDING_SIZE);
    auto U = utility::zeros<matrix_type::upper>(queue, A->shape, A->padding);

    linalg::cholesky_rowwise(queue, A, U);

    for (std::size_t i = 0; i < A->n_rows; ++i) {
        for (std::size_t j = i; j < A->n_cols; ++j) {
            fmt::print("{} ", A(i, j));
        }
        fmt::println("");
    }

    // for (std::size_t i = 0; i < U->n_rows; ++i) {
    //     for (std::size_t j = i; j < U->n_cols; ++j) {
    //         fmt::print("{} ", U(i, j));
    //     }
    //     fmt::println("");
    // }

    EXPECT_NEAR(U(0, 0), 2.0, abs_err);
    EXPECT_NEAR(U(0, 1), 6.0, abs_err);
    EXPECT_NEAR(U(0, 2), -8.0, abs_err);
    EXPECT_NEAR(U(1, 1), 1.0, abs_err);
    EXPECT_NEAR(U(1, 2), 5.0, abs_err);
    EXPECT_NEAR(U(2, 2), 3.0, abs_err);
}

TEST(Linalg, CholeskyRowwiseLarge) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    // crashes with PADDING_SIZE + 1
    auto A = utility::randn<matrix_type::upper>(queue, 33, 33, PADDING_SIZE);
    auto U = utility::zeros<matrix_type::upper>(queue, A->shape, A->padding);

    linalg::cholesky_rowwise(queue, A, U);
}
*/

/*
 * *******************************
 * * TEST
 * *******************************
 */
TEST(TEST, Fill) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const auto N = 16;
    const auto M = 16;

    auto A = utility::zeros<matrix_type::general>(queue, N, M, PADDING_SIZE);

    linalg::blas::fill(queue, A);

    for (auto i = 0; i < N; ++i) {
        for (auto j = 0; j < M; ++j) {
            fmt::print("{} ", A(i, j));
        }
        fmt::println("");
    }
}

/*
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

*/

/*
 * *******************************
 * * (Direct) Cholesky Decomposition
 * *******************************
 */

/*
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
 */

/*
 * *******************************
 * * Jacobi
 * *******************************
 */

/*
TEST(LinearAlgebra, Jacobi) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 4, 12, -16, 37, -43, 98 }, 3, 3, queue);
    auto D = helper::zeros_like(A, queue);

    linalg::jacobi(A, D, queue);

    EXPECT_NEAR(D(0, 0), 1.0 / 4.0, abs_err);
    EXPECT_NEAR(D(1, 1), 1.0 / 37.0, abs_err);
    EXPECT_NEAR(D(2, 2), 1.0 / 98.0, abs_err);

    ::sycl::free(A.data(), queue);
    ::sycl::free(D.data(), queue);
}
*/
