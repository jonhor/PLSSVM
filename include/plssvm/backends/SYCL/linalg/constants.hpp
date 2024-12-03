#ifndef PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_

#include "plssvm/constants.hpp"

#include <cmath>

namespace plssvm::sycl::linalg {

constexpr real_type eps = 1e-10;

constexpr unsigned MAX_BLOCK_SIZE = 32;  // depends on hardware
constexpr unsigned BLOCK_SIZE = std::min(PADDING_SIZE, MAX_BLOCK_SIZE);

}  // namespace plssvm::sycl::linalg
#endif  // PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_
