/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a very small RAII wrapper around a cl_command_queue including information about its associated OpenCL context and device.
 */

#pragma once

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

namespace plssvm::opencl::detail {

/**
 * @brief RAII wrapper class around a cl_command_queue.
 * @details Also contains information about the associated cl_context and cl_device_id.
 */
class command_queue {
  public:
    /**
     * @brief Empty default construct command queue.
     */
    command_queue() = default;
    /**
     * @brief Construct a command queue with the provided information.
     * @param[in] context the associated OpenCL cl_context
     * @param[in] queue the OpenCL cl_command_queue to wrap
     * @param[in] device the associated OpenCL cl_device_id
     */
    command_queue(cl_context context, cl_command_queue queue, cl_device_id device) :
        context{ context }, queue{ queue }, device{ device } {}

    /**
     * @brief Release the cl_command_queue resources on destruction.
     */
    ~command_queue() {
        if (queue) {
            clReleaseCommandQueue(queue);
        }
    }

    /// The OpenCL context associated with the wrapped cl_command_queue.
    cl_context context{};
    /// The wrapped cl_command_queue.
    cl_command_queue queue{};
    /// The OpenCL device associated with the wrapped cl_command_queue.
    cl_device_id device{};
};

}  // namespace plssvm::opencl::detail