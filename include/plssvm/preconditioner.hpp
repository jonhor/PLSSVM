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
    virtual std::chrono::duration<long, std::milli> apply(const soa_matrix<real_type> &B, soa_matrix<real_type> &C) = 0;

    /**
     *
     * Some preconditioners provide a custom product function, calculating Q = A * D
     * This is consistent with the cg function from scipy, where it is possible to provide a LinearOperator instead of A.
     */
    virtual std::chrono::duration<long, std::milli> custom_product(const soa_matrix<real_type> &D, soa_matrix<real_type> &Q) = 0;

    /**
     * This function can be used to query whether the preconditioner provides a custom product function.
     */
    virtual bool has_custom_product() { return false; }

    /**
     * This function can be used to query whether the residuals should be recalculated every X iterations.
     */
    virtual bool recalculate_residuals() { return true; }

    virtual ~preconditioner() { }
};

}  // namespace plssvm

#endif  // PLSSVM_PRECONDITIONER_HPP_
