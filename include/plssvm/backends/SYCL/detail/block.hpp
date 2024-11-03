#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
#pragma once

#include <cstddef>

struct block {
    std::size_t size;
    std::size_t row_offset;
    std::size_t col_offset;
};

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_BLOCK_HPP_
