
#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
#pragma once

#include <random>  // std::random_device, std::default_random_engine

namespace plssvm::sycl::detail {

class rng {
  public:
    rng() :
        random_device_(std::random_device{}),
        random_engine_(random_device_()),
        real_distribution_(0.0, 1.0) {
    }

    real_type randf() {
        return real_distribution_(random_engine_);
    }

  private:
    std::uniform_real_distribution<real_type> real_distribution_;
    std::random_device random_device_;
    std::default_random_engine random_engine_;
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
