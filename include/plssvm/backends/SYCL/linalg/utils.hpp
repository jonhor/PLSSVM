
#ifndef PLSSVM_BACKENDS_SYCL_LINALG_UTILS_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_UTILS_HPP_

#include "constants.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_types.hpp"

namespace plssvm::sycl::linalg {

/*
 * Modified sign function that returns 1 when the value is 0
 */
template <typename T>
T sgn(T val) {
    return val >= 0 ? 1 : -1;
}
}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_UTILS_HPP_
