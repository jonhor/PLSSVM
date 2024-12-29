#ifndef PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_

#include "plssvm/constants.hpp"

#include <cmath>

namespace plssvm::sycl::linalg {

constexpr real_type eps = static_cast<real_type>(1e-10);

// this should be dynamically queried through the queue / device interface
constexpr unsigned MAX_BLOCK_SIZE = 32;  // depends on hardware
constexpr unsigned BLOCK_SIZE = std::min(PADDING_SIZE, MAX_BLOCK_SIZE);
constexpr unsigned MAX_WORKGROUP_SIZE = BLOCK_SIZE * BLOCK_SIZE;

}  // namespace plssvm::sycl::linalg
#endif  // PLSSVM_BACKENDS_SYCL_LINALG_CONSTANTS_HPP_
