#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_UTILITY_HPP_
#pragma once

#include "sycl/sycl.hpp"

#include <cstdlib>

namespace plssvm::sycl::detail {

inline std::size_t align_to(std::size_t value, std::size_t boundary) {
    auto rem = value % boundary;
    if (rem > 0) {
        return value + boundary - rem;
    }
    return value;
}

inline ::sycl::nd_range<1> get_uniform_1d_range(std::size_t n_elements, std::size_t workgroup_size) {
    return ::sycl::nd_range{ ::sycl::range(align_to(n_elements, workgroup_size)), ::sycl::range(workgroup_size) };
}

inline ::sycl::nd_range<2> get_uniform_2d_range(std::size_t n, std::size_t m, std::size_t block_size) {
    auto aligned_n = align_to(n, block_size);
    auto aligned_m = align_to(m, block_size);
    return ::sycl::nd_range{ ::sycl::range(aligned_n, aligned_m), ::sycl::range(block_size, block_size) };
}

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_UTILITY_HPP_
