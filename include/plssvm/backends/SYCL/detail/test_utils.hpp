#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_TEST_UTILS_HPP
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_TEST_UTILS_HPP
#pragma once

#include "plssvm/backends/SYCL/linalg/linalg.hpp"
#include "plssvm/detail/assert.hpp"

#include <fmt/format.h>
#include <string>
#include <string_view>

namespace plssvm::sycl::test {

/**
 * Defines an absolute error that can be used in test functions like ASSERT_NEAR and EXPECT_NEAR
 */
constexpr double abs_err = 1e-6;

/**
 * Defines the path where test data is loaded from
 */
constexpr std::string_view data_path = "/home/jns/dev/thesis/wiki/code/py/data";

[[nodiscard]] inline std::string path(const char *s) {
    return fmt::format("{}/{}", data_path, s);
}

}  // namespace plssvm::sycl::test

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_TEST_UTILS_HPP
