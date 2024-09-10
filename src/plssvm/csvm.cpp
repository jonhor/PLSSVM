/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"

#include "plssvm/constants.hpp"                   // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"              // plssvm::detail::log
#include "plssvm/detail/move_only_any.hpp"        // plssvm::detail::move_only_any
#include "plssvm/detail/operators.hpp"            // plssvm operator overloads for vectors
#include "plssvm/detail/performance_tracker.hpp"  // PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking_entry
#include "plssvm/detail/utility.hpp"              // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::invalid_parameter_exception
#include "plssvm/gamma.hpp"                       // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"            // plssvm::kernel_function
#include "plssvm/matrix.hpp"                      // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/shape.hpp"                       // plssvm::shape
#include "plssvm/solver_types.hpp"                // plssvm::solver_type
#include "plssvm/verbosity_levels.hpp"            // plssvm::verbosity_level

#include "fmt/core.h"  // fmt::format

#include <algorithm>   // std::count
#include <chrono>      // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>     // std::size_t
#include <functional>  // std::plus
#include <numeric>     // std::inner_product
#include <utility>     // std::move
#include <utility>     // std::pair, std::make_pair
#include <variant>     // std::holds_alternative, std::get
#include <vector>      // std::vector

namespace plssvm {

void csvm::sanity_check_parameter() const {
    // kernel: valid kernel function
    const auto kernel_type_value = detail::to_underlying(params_.kernel_type);
    if (kernel_type_value < 0 || kernel_type_value >= 6) {
        throw invalid_parameter_exception{ fmt::format("Invalid kernel function with value {} given!", kernel_type_value) };
    }

    // gamma: must be greater than 0 IF explicitly provided as real_type (not for the linear kernel)
    if (params_.kernel_type != kernel_function_type::linear && std::holds_alternative<real_type>(params_.gamma) && std::get<real_type>(params_.gamma) <= real_type{ 0.0 }) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", std::get<real_type>(params_.gamma)) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

std::pair<soa_matrix<real_type>, unsigned long long> csvm::conjugate_gradients(const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, const std::optional<std::vector<detail::move_only_any>> &M, const real_type eps, const unsigned long long max_cg_iter, const solver_type cg_solver) const {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(!B.empty(), "The right-hand sides must not be empty!");
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The epsilon value must be greater than 0.0!");
    PLSSVM_ASSERT(max_cg_iter > 0, "The maximum number of iterations must be greater than 0!");

    const std::size_t num_rows = B.num_cols();
    const std::size_t num_rhs = B.num_rows();

    // timing for each CG iteration
    std::chrono::milliseconds total_iteration_time{};
    std::chrono::milliseconds total_blas_level_3_time{};

    //
    // perform Conjugate Gradients (CG) algorithm
    //

    soa_matrix<real_type> X{ shape{ num_rhs, num_rows }, real_type{ 1.0 }, shape{ PADDING_SIZE, PADDING_SIZE } };

    // R = B - A * X
    soa_matrix<real_type> R{ B, shape{ PADDING_SIZE, PADDING_SIZE } };
    total_blas_level_3_time += this->run_blas_level_3(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R);

    // if M: D = M * R
    // else: D = R
    soa_matrix<real_type> D{ R, shape{ PADDING_SIZE, PADDING_SIZE } };
    if (M.has_value()) {
        total_blas_level_3_time += this->run_blas_level_3(cg_solver, real_type{ 1.0 }, *M, R, real_type{ 0.0 }, D);
    }

    // delta = R.T * D
    std::vector<real_type> delta = rowwise_dot(R, D);
    const std::vector<real_type> delta0(delta);

    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto rhs_idx_max_residual_difference = [&]() {
        const real_type max_difference{ 0.0 };
        std::size_t idx{ 0 };
        for (std::size_t i = 0; i < delta.size(); ++i) {
            const real_type difference = delta[i] - (eps * eps * delta0[i]);
            if (difference > max_difference) {
                idx = i;
            }
        }
        return idx;
    };
    // get the number of rhs that have already been converged
    const auto num_rhs_converged = [eps, &delta, &delta0]() {
        return static_cast<std::size_t>(std::inner_product(delta.cbegin(), delta.cend(), delta0.cbegin(), real_type{ 0.0 }, std::plus<>{}, [eps](const real_type d, const real_type d0) { return d <= eps * eps * d0; }));
    };
    // calculate a mask for every converged right hand side
    // -> 0 if the rhs already converged, 1 otherwise
    const auto calculate_rhs_converged_mask = [eps, &delta, &delta0]() {
        std::vector<int> mask(delta.size());
        for (std::size_t i = 0; i < mask.size(); ++i) {
            mask[i] = delta[i] <= eps * eps * delta0[i] ? 0 : 1;
        }
        return mask;
    };

    unsigned long long iter = 0;
    while (iter < max_cg_iter && num_rhs_converged() < num_rhs) {
        const std::size_t max_residual_difference_idx = rhs_idx_max_residual_difference();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_cg_iter,
                    num_rhs_converged(),
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        const std::chrono::steady_clock::time_point iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        soa_matrix<real_type> Q{ shape{ D.num_rows(), D.num_cols() }, shape{ PADDING_SIZE, PADDING_SIZE } };
        total_blas_level_3_time += this->run_blas_level_3(cg_solver, real_type{ 1.0 }, A, D, real_type{ 0.0 }, Q);

        // alpha = delta_new / (D^T * Q))
        const std::vector<real_type> alpha = delta / rowwise_dot(D, Q);

        // create mask for the residual -> only update X if the respective rhs did not already converge
        std::vector<int> mask = calculate_rhs_converged_mask();
        // update mask -> can't update X if the residual is already 0
#pragma omp parallel for shared(R, mask)
        for (std::size_t row = 0; row < R.num_rows(); ++row) {
            // if mask[row] == 0 -> rhs already converged -> X wouldn't be updated either way
            if (mask[row] == 1) {
                // check if any residual value is not zero
                bool is_residual_zero = true;
                for (std::size_t col = 0; col < R.num_cols(); ++col) {
                    if (R(row, col) != real_type{ 0.0 }) {
                        is_residual_zero = false;
                        break;
                    }
                }
                // all residual values are 0 -> residual is 0 -> can't updated X for this rhs!
                if (is_residual_zero) {
                    mask[row] = 0;
                }
            }
        }

        // X = X + alpha * D
        X += masked_rowwise_scale(mask, alpha, D);

        if (iter % 50 == 49) {
            // explicitly recalculate residual to remove accumulating floating point errors
            // R = B - A * X
            R = soa_matrix<real_type>{ B, shape{ PADDING_SIZE, PADDING_SIZE } };
            total_blas_level_3_time += this->run_blas_level_3(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R);
        } else {
            // R = R - alpha * Q
            R -= rowwise_scale(alpha, Q);
        }

        // delta_old = delta_new
        const std::vector<real_type> delta_old = delta;

        // if M: delta_new = R.T * S, where S = M * R
        // else: delta_new = R.T * R
        soa_matrix<real_type> S{ D.shape(), D.padding() };
        if (M.has_value()) {
            total_blas_level_3_time += this->run_blas_level_3(cg_solver, real_type{ 1.0 }, *M, R, real_type{ 0.0 }, S);
            delta = rowwise_dot(R, S);
        } else {
            delta = rowwise_dot(R, R);
        }

        // beta = delta_new / delta_old
        const std::vector<real_type> beta = delta / delta_old;

        // if M: D = beta * D + S
        // else: D = beta * D + R
        if (M.has_value()) {
            D = rowwise_scale(beta, D) + S;
        } else {
            D = rowwise_scale(beta, D) + R;
        }

        const std::chrono::steady_clock::time_point iteration_end_time = std::chrono::steady_clock::now();
        const std::chrono::duration iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        total_iteration_time += iteration_duration;

        // next CG iteration
        ++iter;
    }

    const std::size_t max_residual_difference_idx = rhs_idx_max_residual_difference();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {} and an average SYMM time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", iter },
                detail::tracking_entry{ "cg", "max_iterations", max_cg_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_rhs_converged() },
                detail::tracking_entry{ "cg", "num_rhs", num_rhs },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking_entry{ "cg", "avg_iteration_time", total_iteration_time / std::max(iter, 1ULL) },
                detail::tracking_entry{ "cg", "avg_blas_level_3_time", total_blas_level_3_time / (1 + iter + iter / 50) });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                iter);

    return std::make_pair(X, iter);
}

std::pair<std::vector<real_type>, real_type> csvm::perform_dimensional_reduction(const parameter &params, const soa_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix must not be empty!");

    const std::chrono::steady_clock::time_point dimension_reduction_start_time = std::chrono::steady_clock::now();

    const std::size_t num_rows_reduced = A.num_rows() - 1;

    // create q_red vector and calculate QA_costs
    std::vector<real_type> q_red(num_rows_reduced);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
#pragma omp parallel for default(none) shared(q_red, A) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::linear>(A, i, A, num_rows_reduced);
            }
            break;
        case kernel_function_type::polynomial:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::polynomial>(A, i, A, num_rows_reduced, params.degree, std::get<real_type>(params.gamma), params.coef0);
            }
            break;
        case kernel_function_type::rbf:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::rbf>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
        case kernel_function_type::sigmoid:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::sigmoid>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma), params.coef0);
            }
            break;
        case kernel_function_type::laplacian:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::laplacian>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
        case kernel_function_type::chi_squared:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::chi_squared>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
    }
    const real_type QA_cost = kernel_function(A, num_rows_reduced, A, num_rows_reduced, params) + real_type{ 1.0 } / params.cost;
    const std::chrono::steady_clock::time_point dimension_reduction_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Performed dimensional reduction in {}.\n",
                detail::tracking_entry{ "cg", "dimensional_reduction", std::chrono::duration_cast<std::chrono::milliseconds>(dimension_reduction_end_time - dimension_reduction_start_time) });

    return std::make_pair(std::move(q_red), QA_cost);
}

std::chrono::duration<long, std::milli> csvm::run_blas_level_3(const solver_type cg_solver, const real_type alpha, const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix must not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix must not be empty!");

    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    this->blas_level_3(cg_solver, alpha, A, B, beta, C);

    const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
}

aos_matrix<real_type> csvm::run_predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    decltype(auto) res = this->predict_values(params, support_vectors, alpha, rho, w, predict_points);

    const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Predicted the values of {} predict points using {} support vectors with {} features each in {}.\n",
                predict_points.num_rows(),
                support_vectors.num_rows(),
                support_vectors.num_cols(),
                detail::tracking_entry{ "predict_values", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    return res;
}

}  // namespace plssvm
