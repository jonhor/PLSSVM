/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Jonas Horstmann
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible preconditioner algorithms to be used.
 */

#ifndef PLSSVM_PRECONDITIONER_TYPES_HPP_
#define PLSSVM_PRECONDITIONER_TYPES_HPP_
#pragma once

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm {

/**
 * @brief Enum class for all possible preconditioner types.
 */
enum class preconditioner_type {
    /**
     * @brief The default preconditioner.
     * @details No preconditioner is used by default.
     */
    none,
    /** Use the jacobi (diagonal) preconditioner. */
    jacobi
};

/**
 * @brief Output the @p preconditioning type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the preconditioner type to
 * @param[in] preconditioning the preconditioner type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, preconditioner_type preconditioning);

/**
 * @brief Use the input-stream @p in to initialize the @p preconditioner type.
 * @param[in,out] in input-stream to extract the preconditioner type from
 * @param[in] preconditioning the preconditioner type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, preconditioner_type &preconditioning);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::preconditioner_type> : fmt::ostream_formatter { };

#endif  // PLSSVM_PRECONDITIONER_TYPES_HPP_
