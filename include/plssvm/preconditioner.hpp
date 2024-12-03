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

    virtual void custom_product() { }

    virtual ~preconditioner() { }
};

}  // namespace plssvm

#endif  // PLSSVM_PRECONDITIONER_HPP_
