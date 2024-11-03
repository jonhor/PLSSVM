#include "plssvm/backends/SYCL/detail/block.hpp"
#include "plssvm/backends/SYCL/detail/linalg.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace plssvm;
using namespace plssvm::sycl::detail;

#define abs_err 1e-6

std::vector<double> read_data_from_file(const char *file_name) {
    // open file
    std::streampos file_size;
    std::ifstream file(file_name, std::ios::binary);

    // get file size
    file.seekg(0, std::ios::end);
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // read data
    std::vector<double> file_data(file_size / sizeof(double));
    file.read((char *) &file_data[0], file_size);
    return file_data;
}

TEST(Cholesky, BasicTest) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto K_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/kernel_matrix.bytes");
    auto K = utility::create_managed_view<matrix_type::upper>(queue, K_vec.data(), 4, 4, PADDING_SIZE);

    auto U_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/upper_cholesky.bytes");
    auto U_solution = utility::create_managed_view<matrix_type::upper>(queue, U_vec.data(), 4, 4, PADDING_SIZE);

    auto U = utility::zeros<matrix_type::upper>(queue, K->shape, K->padding);
    linalg::chol(queue, K, U);

    for (std::size_t i = 0; i < K->n_rows; ++i) {
        for (std::size_t j = i; j < K->n_cols; ++j) {
            ASSERT_NEAR(U(i, j), U_solution(i, j), abs_err);
        }
    }
}

TEST(Cholesky, Matrix1000) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 1000;

    auto K_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/kernel_matrix_1000.bytes");
    auto K = utility::create_managed_view<matrix_type::upper>(queue, K_vec.data(), N, N, PADDING_SIZE);

    auto U_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/upper_cholesky_1000.bytes");
    auto U_solution = utility::create_managed_view<matrix_type::upper>(queue, U_vec.data(), N, N, PADDING_SIZE);

    auto U = utility::zeros<matrix_type::upper>(queue, K->shape, K->padding);
    linalg::chol(queue, K, U);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = i; j < N; ++j) {
            ASSERT_NEAR(U(i, j), U_solution(i, j), abs_err);
        }
    }
}

//
// TEST(TRSM, Matrix1000) {
//    ::sycl::default_selector selector;
//    ::sycl::queue queue{ selector };
//
//    const std::size_t N = 1000;
//    const std::size_t M = 8;
//
//    auto A_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_a_1000.bytes");
//    auto A = utility::create_managed_view<matrix_type::upper>(queue, A_vec.data(), N, N, PADDING_SIZE);
//
//    auto B_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_b_1000.bytes");
//    auto B = utility::create_managed_view<matrix_type::general>(queue, B_vec.data(), N, M, PADDING_SIZE);
//
//    auto X_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_x_1000.bytes");
//    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, X_vec.data(), N, M, PADDING_SIZE);
//
//    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
//    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), 10);
//
//    for (std::size_t i = 0; i < N; ++i) {
//        for (std::size_t j = 0; j < M; ++j) {
//            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
//        }
//    }
//}

TEST(TRSM, LowerMatrix50) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 50;
    const std::size_t M = 8;

    auto A_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_lower_a_50.bytes");
    auto A = utility::create_managed_view<matrix_type::lower>(queue, A_vec.data(), N, N, PADDING_SIZE);

    auto B_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_lower_b_50.bytes");
    auto B = utility::create_managed_view<matrix_type::general>(queue, B_vec.data(), N, M, PADDING_SIZE);

    auto X_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_lower_x_50.bytes");
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, X_vec.data(), N, M, PADDING_SIZE);

    linalg::blas::triangular_solve_lower(queue, A.view(), B.view());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(B(i, j), X_solution(i, j), abs_err);
        }
    }
}

TEST(TRSM, LowerBlocksRemainderColumnsNotExact) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 5;
    const std::size_t M = 2;
    const std::size_t block_size = 4;

    auto A = utility::create_managed_view<matrix_type::lower>(queue, { 1, 2, 1, 1, 3, 2, 3, 2, 1, 2, 4, 1, 3, 2, 4 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 3, 4, 3, 2, 4 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 1, 0, 0.5, 0.5, -0.75, -1.75, -0.75, -0.5 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::triangular_solve_lower(queue, A.view(), B.view());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", B(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(B(i, j), X_solution(i, j), abs_err);
        }
    }
}

/*
TEST(TRSM, Matrix50) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 50;
    const std::size_t M = 6;

    auto A_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_a_50.bytes");
    auto A = utility::create_managed_view<matrix_type::upper>(queue, A_vec.data(), N, N, PADDING_SIZE);

    auto B_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_b_50.bytes");
    auto B = utility::create_managed_view<matrix_type::general>(queue, B_vec.data(), N, M, PADDING_SIZE);

    auto X_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_x_50.bytes");
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, X_vec.data(), N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::triangular_solve(queue, A.view(), B.view());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::println("({}, {})", i, j);
            ASSERT_NEAR(B(i, j), X_solution(i, j), abs_err);
        }
    }
}
 */

TEST(TRSM, UpperBlocksRemainder) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 5;
    const std::size_t M = 2;
    const std::size_t block_size = 2;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 2, 1, 3, 4, 1, 3, 2, 1, 2, 1, 3, 2, 2, 4 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 3, 4, 3, 2, 4 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 0.5, -8.75, -3.5, 2.75, 1, -0.25, 1.5, 0.5, 0.5, 1 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), block_size);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
        }
    }
}

TEST(TRSM, UpperBlocksRemainderColumnsNotExact) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 5;
    const std::size_t M = 2;
    const std::size_t block_size = 2;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 2, 1, 3, 4, 1, 3, 2, 1, 2, 1, 3, 2, 2, 4 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 3, 4, 3, 2, 4 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 0.5, -8.75, -3.5, 2.75, 1, -0.25, 1.5, 0.5, 0.5, 1 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), block_size);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
        }
    }
}

TEST(TRSM, UpperBasic) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 3;
    const std::size_t M = 1;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 2, 1, 1, 3, 2 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 4.5, -2.5, 1.5 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::trsm(queue, A.view(), B.view(), X.view());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
        }
    }
}

/* triangular solve sheninagans */
TEST(TRSM, UpperTriangularSolveBasic) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 3;
    const std::size_t M = 1;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 2, 1, 1, 3, 2 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 4.5, -2.5, 1.5 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::triangular_solve(queue, A.view(), B.view());

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", B(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(B(i, j), X_solution(i, j), abs_err);
        }
    }
}

TEST(TRSM, UpperhjkhjBlocksRemainderColumnsNotExact) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 5;
    const std::size_t M = 2;
    const std::size_t block_size = 2;

    auto A = utility::create_managed_view<matrix_type::upper>(queue, { 1, 2, 1, 3, 4, 1, 3, 2, 1, 2, 1, 3, 2, 2, 4 }, N, N, PADDING_SIZE);
    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 3, 4, 3, 2, 4 }, N, M, PADDING_SIZE);
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 0.5, -8.75, -3.5, 2.75, 1, -0.25, 1.5, 0.5, 0.5, 1 }, N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::triangular_solve(queue, A.view(), B.view(), block_size);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", B(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(B(i, j), X_solution(i, j), abs_err);
        }
    }
}

/*
TEST(TRSM, UpperBasic2) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    const std::size_t N = 4;
    const std::size_t M = 2;

    auto A_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_a.bytes");
    auto A = utility::create_managed_view<matrix_type::upper>(queue, A_vec.data(), N, N, PADDING_SIZE);

    auto B_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_b.bytes");
    auto B = utility::create_managed_view<matrix_type::general>(queue, B_vec.data(), N, M, PADDING_SIZE);

    auto X_vec = read_data_from_file("/home/jns/dev/thesis/wiki/code/py/data/trsm_upper_x.bytes");
    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, X_vec.data(), N, M, PADDING_SIZE);

    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), 2);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            fmt::print("{} ", X(i, j));
        }
        fmt::println("");
    }

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
        }
    }
}
 */

//
// TEST(TRSM, LowerBlocksRemainderColumnsNotExact) {
//    ::sycl::default_selector selector;
//    ::sycl::queue queue{ selector };
//
//    const std::size_t N = 5;
//    const std::size_t M = 2;
//    const std::size_t block_size = 4;
//
//    auto A = utility::create_managed_view<matrix_type::lower>(queue, { 1, 2, 1, 1, 3, 2, 3, 2, 1, 2, 4, 1, 3, 2, 4 }, N, N, PADDING_SIZE);
//    auto B = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 3, 4, 5, 3, 4, 3, 2, 4 }, N, M, PADDING_SIZE);
//    auto X_solution = utility::create_managed_view<matrix_type::general>(queue, { 1, 2, 1, 0, 0.5, 0.5, -0.75, -1.75, -0.75, -0.5 }, N, M, PADDING_SIZE);
//
//    auto X = utility::zeros<matrix_type::general>(queue, B->shape, B->padding);
//    linalg::blas::trsm(queue, A.view(), B.view(), X.view(), block_size);
//
//    for (std::size_t i = 0; i < N; ++i) {
//        for (std::size_t j = 0; j < M; ++j) {
//            fmt::print("{} ", X(i, j));
//        }
//        fmt::println("");
//    }
//
//    for (std::size_t i = 0; i < N; ++i) {
//        for (std::size_t j = 0; j < M; ++j) {
//            ASSERT_NEAR(X(i, j), X_solution(i, j), abs_err);
//        }
//    }
//}
