#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
#pragma once

#include <cstddef>

namespace plssvm::sycl::detail {

struct block {
    std::size_t size;
    std::size_t row_offset;
    std::size_t col_offset;
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
