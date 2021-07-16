/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements conversion functions from arithmetic types to their name as string representation.
 */

#pragma once

#include <string_view>  // std::string_view

/**
 * @def PLSSVM_CREATE_ARITHMETIC_TYPE_NAME
 * @brief Defines a macro to create all possible conversion functions from arithmetic types to their name as string representation.
 * @param[in] type the data type to convert to a string
 */
#define PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(type) \
    template <>                                  \
    [[nodiscard]] constexpr inline std::string_view arithmetic_type_name<type>() { return #type; }

namespace plssvm::detail {

/**
 * @brief Tries to convert the given type to its name as string representation.
 * @details The definition is marked as **deleted** if `T` isn't an arithmetic type (without cvref-qualification).
 * @tparam T the type to convert to a string
 * @return the name of `T`
 */
template <typename T>
[[nodiscard]] constexpr inline std::string_view arithmetic_type_name() = delete;

PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(bool)

// character types
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(char)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(signed char)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(unsigned char)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(char16_t)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(char32_t)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(wchar_t)

// integer types
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(short)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(unsigned short)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(int)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(unsigned int)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(long)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(unsigned long)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(long long)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(unsigned long long)

// floating point types
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(float)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(double)
PLSSVM_CREATE_ARITHMETIC_TYPE_NAME(long double)

}  // namespace plssvm::detail

#undef PLSSVM_CREATE_ARITHMETIC_TYPE_NAME