/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/backends/SYCL/implementation_type.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::sycl_generic {

std::ostream &operator<<(std::ostream &out, const implementation_type impl) {
    switch (impl) {
        case implementation_type::automatic:
            return out << "automatic";
        case implementation_type::dpcpp:
            return out << "dpcpp";
        case implementation_type::hipsycl:
            return out << "hipsycl";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, implementation_type &impl) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        impl = implementation_type::automatic;
    } else if (str == "dpcpp" || str == "dpc++") {
        impl = implementation_type::dpcpp;
    } else if (str == "hipsycl") {
        impl = implementation_type::hipsycl;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::sycl_generic