/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/preconditioner_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const preconditioner_type preconditioning) {
    switch (preconditioning) {
        case preconditioner_type::none:
            return out << "none";
        case preconditioner_type::jacobi:
            return out << "jacobi";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, preconditioner_type &preconditioning) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "none") {
        preconditioning = preconditioner_type::none;
    } else if (str == "jacobi") {
        preconditioning = preconditioner_type::jacobi;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
