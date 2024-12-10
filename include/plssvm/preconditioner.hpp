#ifndef PLSSVM_PRECONDITIONER_HPP_
#define PLSSVM_PRECONDITIONER_HPP_
#pragma once

#include "plssvm/constants.hpp"
#include "plssvm/matrix.hpp"

namespace plssvm {

class preconditioner {
  public:
    /**
     * Apply the precondition matrix by calculating C = M * B
     */
    virtual void apply(const soa_matrix<real_type> &B, soa_matrix<real_type> &C) = 0;

    /**
     *
     * Some preconditioners provide a custom product function, calculating Q = A * D
     * This is consistent with the cg function from scipy, where it is possible to provide a LinearOperator instead of A.
     */
    virtual void custom_product(const soa_matrix<real_type> &D, soa_matrix<real_type> &Q) { }

    /**
     * This function can be used to query whether the preconditioner provides a custom product function.
     * This is not the cleanest design but will do for now.
     */
    virtual bool has_custom_product() { return false; }

    virtual ~preconditioner() { }
};

}  // namespace plssvm

#endif  // PLSSVM_PRECONDITIONER_HPP_
