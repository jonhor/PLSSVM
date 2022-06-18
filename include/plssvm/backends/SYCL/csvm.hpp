/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the SYCL backend.
 */

#pragma once

#include "plssvm/kernel_types.hpp"                                                  // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform
#include "plssvm/parameter.hpp"                                                     // plssvm::parameter
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp"  // forward declaration and namespace alias
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/device_ptr.hpp" // plssvm::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::detail::device_ptr
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"                          // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/backends/gpu_csvm.hpp"                                             // plssvm::detail::gpu_csvm

#include <memory>   // std::unique_ptr
#include <utility>  // std::forward

namespace plssvm {

using namespace sycl_generic;

namespace detail {

// forward declare execution_range class
class execution_range;

}  // namespace detail

namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@ {

/**
 * @brief A C-SVM implementation using SYCL as backend.
 * @details If DPC++ is available, this class also exists in the `plssvm::dpcpp` namespace.
 *          If hipSYCL is available, this class also exists in the `plssvm::hipsycl` namespace.
 * @tparam T the type of the data
 */
template <typename T>
    class csvm : public ::plssvm::detail::gpu_csvm<T, detail::device_ptr<T>, std::unique_ptr<detail::sycl::queue>> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the SYCL C-SVM class.
    using base_type = ::plssvm::detail::gpu_csvm<T, detail::device_ptr<T>, std::unique_ptr<detail::sycl::queue>>;

    using base_type::devices_;

  public:
    using typename base_type::real_type;
    using typename base_type::size_type;
    using typename base_type::device_ptr_type;
    using typename base_type::queue_type;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::sycl::backend_exception if the requested plssvm::target_platform isn't available
     * @throws plssvm::sycl::backend_exception if no possible OpenCL devices could be found
     */
    explicit csvm(target_platform target, parameter<real_type> params = {});

    template <typename... Args>
    csvm(target_platform target, kernel_type kernel, Args&&... named_args) : base_type{ kernel, std::forward<Args>(named_args)... } {
        // TODO: additional parameter?!?!!!
        this->init(target, kernel);
    }

    /**
     * @brief Wait for all operations in all [`sycl::queue`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:interface.queue.class) to finish.
     * @details Terminates the program, if any asynchronous exception is thrown.
     */
    ~csvm() override;

  protected:
    /**
     * @copydoc plssvm::detail::gpu_csvm::device_synchronize
     */
    void device_synchronize(queue_type &queue) const final;

    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    void run_q_kernel(size_type device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &q_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points_padded, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    void run_svm_kernel(size_type device, const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, real_type QA_cost, real_type add, size_type num_data_points_padded, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_w_kernel
     */
    void run_w_kernel(size_type device, [[maybe_unused]] const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_data_points, size_type num_features) const final;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_predict_kernel
     */
    void run_predict_kernel(const ::plssvm::detail::execution_range &range, const parameter<real_type> &params, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const device_ptr_type &data_d, const device_ptr_type &data_last_d, size_type num_support_vectors, size_type num_predict_points, size_type num_features) const final;

  private:
    void init(target_platform target);

    /// The SYCL kernel invocation type for the svm kernel. Either nd_range or hierarchical.
    kernel_invocation_type invocation_type_;
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace @PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@
}  // namespace plssvm