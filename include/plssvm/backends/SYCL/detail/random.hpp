#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
#pragma once

#include "plssvm/constants.hpp"

#include <random>  // std::random_device, std::default_random_engine, std::uniform_real_distribution, std::discrete_distribution

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

    template <class IntType = int>
    IntType randint(IntType begin, IntType end) {
        std::uniform_int_distribution distribution(begin, end);
        return distribution(random_engine_);
    }

    template <class IntType = int, class InputIt>
    IntType choice(InputIt begin, InputIt end) {
        std::discrete_distribution<IntType> distribution(begin, end);
        return distribution(random_engine_);
    }

  private:
    std::random_device random_device_;
    std::default_random_engine random_engine_;
    std::uniform_real_distribution<real_type> real_distribution_;
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_RANDOM_HPP_
