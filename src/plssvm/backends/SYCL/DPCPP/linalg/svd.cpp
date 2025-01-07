#include "plssvm/backends/SYCL/linalg/decompositions/svd.hpp"

#include "plssvm/backends/SYCL/linalg/matrix/common.hpp"
#include "plssvm/backends/SYCL/linalg/matrix/matrix.hpp"

#include "sycl/sycl.hpp"

#include "oneapi/mkl.hpp"
#include <cstdint>

namespace plssvm::sycl::linalg {

svd_return_type svd(::sycl::queue &queue, const matrix_view<matrix_type::general> &A) {
    using namespace oneapi::mkl;

    // gesvd from oneMKL is only supported on CPU
    ::sycl::queue cpu_queue{ ::sycl::cpu_selector_v };

    // gesvd expects the matrix to be in column-major order
    auto AT = linalg::transposed(queue, A);
    const auto m = static_cast<std::int64_t>(A.n_rows);
    const auto n = static_cast<std::int64_t>(A.n_cols);
    const auto p = static_cast<std::int64_t>(A.padding);

    auto UT = linalg::empty<matrix_type::general>(queue, AT->n_rows, AT->n_cols, AT->padding);
    auto d = static_cast<std::size_t>(std::min(m, n));
    auto S = linalg::empty<matrix_type::diagonal>(queue, d, d, AT->padding);

    // calculate U as thin SVD
    auto job_u = jobsvd::somevec;

    // skip calculating VT
    auto job_vt = jobsvd::novec;

    // leading dimensions (ld) include padding
    const auto ld_a = m + p;
    const auto ld_u = m + p;
    const auto ld_vt = m + p;

    // allocate scratchpad
    auto scratchpad_size = lapack::gesvd_scratchpad_size<real_type>(queue, job_u, job_vt, m, n, ld_a, ld_u, ld_vt);
    auto scratchpad = ::sycl::malloc_shared<real_type>(static_cast<std::size_t>(scratchpad_size), queue);

    // calculate thin SVD
    auto event = lapack::gesvd(cpu_queue, job_u, job_vt, m, n, AT->data(), ld_a, S->data(), UT->data(), ld_u, nullptr, ld_vt, scratchpad, scratchpad_size);
    event.wait();

    auto U = linalg::transposed(queue, UT);

    return std::make_tuple(std::move(U), std::move(S));
}

}  // namespace plssvm::sycl::linalg
