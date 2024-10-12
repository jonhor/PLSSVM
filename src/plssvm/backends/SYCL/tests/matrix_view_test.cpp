#include "plssvm/backends/SYCL/detail/matrix_view.hpp"

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace plssvm;
using namespace plssvm::sycl::detail;

/*
 * *******************************
 * * General Matrix
 * *******************************
 */
TEST(MatrixView, General_BasicIndexing) {
    std::vector<real_type> elems{ 1, 2, 3, 4 };
    matrix_view<matrix_type::general> A(elems.data(), 2, 2);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(1, 0), 3);
    EXPECT_EQ(A(1, 1), 4);
}

TEST(MatrixView, General_PaddedIndexing) {
    std::vector<real_type> elems{ 1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 5, 6 };
    matrix_view<matrix_type::general> A(elems.data(), 2, 2, 3);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(1, 0), 3);
    EXPECT_EQ(A(1, 1), 4);
    EXPECT_EQ(A(2, 0), 5);
    EXPECT_EQ(A(2, 1), 6);
}

/*
 * *******************************
 * * Upper Triangular Matrix
 * *******************************
 */
TEST(MatrixView, Upper_BasicIndexing) {
    std::vector<real_type> elems{ 1, 2, 3, 4, 5, 6 };
    matrix_view<matrix_type::upper> A(elems.data(), 3);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(0, 2), 3);
    EXPECT_EQ(A(1, 1), 4);
    EXPECT_EQ(A(1, 2), 5);
    EXPECT_EQ(A(2, 2), 6);
}

TEST(MatrixView, Upper_PaddedIndexing) {
    std::vector<real_type> elems{ 1, 2, 3, 0, 0, 4, 5, 0, 0, 6, 0, 0, 0, 0, 0 };
    matrix_view<matrix_type::upper> A(elems.data(), 3, 3, 2);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(0, 1), 2);
    EXPECT_EQ(A(0, 2), 3);
    EXPECT_EQ(A(1, 1), 4);
    EXPECT_EQ(A(1, 2), 5);
    EXPECT_EQ(A(2, 2), 6);
}

/*
 * *******************************
 * * Lower Triangular Matrix
 * *******************************
 */
TEST(MatrixView, Lower_BasicIndexing) {
    std::vector<real_type> elems = { 1, 2, 3, 4, 5, 6 };
    matrix_view<matrix_type::lower> A(elems.data(), 3);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(1, 0), 2);
    EXPECT_EQ(A(1, 1), 3);
    EXPECT_EQ(A(2, 0), 4);
    EXPECT_EQ(A(2, 1), 5);
    EXPECT_EQ(A(2, 2), 6);
}

TEST(MatrixView, Lower_PaddedIndexing) {
    std::vector<real_type> elems = { 1, 0, 0, 2, 3, 0, 0, 4, 5, 6 };
    matrix_view<matrix_type::lower> A(elems.data(), 3, 3, 2);

    EXPECT_EQ(A(0, 0), 1);
    EXPECT_EQ(A(1, 0), 2);
    EXPECT_EQ(A(1, 1), 3);
    EXPECT_EQ(A(2, 0), 4);
    EXPECT_EQ(A(2, 1), 5);
    EXPECT_EQ(A(2, 2), 6);
}

/*
 * *******************************
 * * Shared Functionality
 * *******************************
 */
TEST(HelperFunctions, CreateSharedView) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::lower>({ 5, 3, 9, 7, 4, 2 }, 3, 3, queue);

    EXPECT_EQ(A(0, 0), 5);
    EXPECT_EQ(A(1, 0), 3);
    EXPECT_EQ(A(1, 1), 9);
    EXPECT_EQ(A(2, 0), 7);
    EXPECT_EQ(A(2, 1), 4);
    EXPECT_EQ(A(2, 2), 2);

    ::sycl::free(A.data(), queue);
}

TEST(HelperFunctions, Transpose_FromLower) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::lower>({ 5, 3, 9, 7, 4, 2 }, 3, 3, queue);
    auto AT = helper::transpose(A, queue);

    EXPECT_EQ(AT(0, 0), 5);
    EXPECT_EQ(AT(0, 1), 3);
    EXPECT_EQ(AT(0, 2), 7);
    EXPECT_EQ(AT(1, 1), 9);
    EXPECT_EQ(AT(1, 2), 4);
    EXPECT_EQ(AT(2, 2), 2);

    ::sycl::free(A.data(), queue);
    ::sycl::free(AT.data(), queue);
}

TEST(HelperFunctions, Transpose_FromUpper) {
    ::sycl::default_selector selector;
    ::sycl::queue queue{ selector };

    auto A = helper::create_shared_view<matrix_type::upper>({ 5, 3, 9, 7, 4, 2 }, 3, 3, queue);
    auto AT = helper::transpose(A, queue);

    EXPECT_EQ(AT(0, 0), 5);
    EXPECT_EQ(AT(1, 0), 3);
    EXPECT_EQ(AT(1, 1), 7);
    EXPECT_EQ(AT(2, 0), 9);
    EXPECT_EQ(AT(2, 1), 4);
    EXPECT_EQ(AT(2, 2), 2);

    ::sycl::free(A.data(), queue);
    ::sycl::free(AT.data(), queue);
}
