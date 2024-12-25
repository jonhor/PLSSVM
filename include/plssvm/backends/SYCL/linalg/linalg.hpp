#ifndef PLSSVM_BACKENDS_SYCL_LINALG_LINALG_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_LINALG_HPP_

#include "constants.hpp"

// matrix abstraction
#include "plssvm/backends/SYCL/linalg/matrix/common.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix_types.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix_view.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/utility.hpp"

// operations
#include "plssvm/backends/SYCL/linalg/operations/arithmetic.hpp"
#include "plssvm/backends/SYCL/linalg/operations/inverse.hpp"
#include "plssvm/backends/SYCL/linalg/operations/triangular_solve.hpp"

// decompositions
#include "plssvm/backends/SYCL/linalg/decompositions/cholesky.hpp"
#include "plssvm/backends/SYCL/linalg/decompositions/rpcholesky.hpp"
#include "plssvm/backends/SYCL/linalg/decompositions/svd.hpp"

// other
//#include "plssvm/backends/SYCL/linalg/utils.hpp"

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_LINALG_HPP_
