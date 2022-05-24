/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#pragma once

#include "plssvm/kernel_types.hpp"      // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/model.hpp"             // plssvm::model
#include "plssvm/data_set.hpp"          // plssvm::data_set

#include <cstddef>      // std::size_t
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector


#include <iostream>
#include <memory>
#include <optional>
#include <functional>
#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl_generic::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/detail/utility.hpp"

namespace plssvm {

// forward declare class TODO
//template <typename T>
//class parameter;

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the type of the data
 */
template <typename T>//, typename U = int>
class csvm {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");
//    static_assert(std::is_arithmetic_v<U> || std::is_same_v<U, std::string>, "The second template type can only be an arithmetic type or 'std::string'!");
//    // because std::vector<bool> is evil
//    static_assert(!std::is_same_v<U, bool>, "The second template type must NOT be 'bool'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
//    using label_type = U;
    using size_type = std::size_t;

    //*************************************************************************************************************************************//
    //                                                      special member functions                                                       //
    //*************************************************************************************************************************************//
//    explicit csvm(const parameter<T> &params);

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Delete expensive copy-constructor to make csvm a move only type.
     */
    csvm(const csvm &) = delete;
    /**
     * @brief Move-constructor as csvm is a move-only type.
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @brief Delete expensive copy-assignment-operator to make csvm a move only type.
     */
    csvm &operator=(const csvm &) = delete;
    /**
     * @brief Move-assignment-operator as csvm is a move-only type.
     * @return `*this`
     */
    csvm &operator=(csvm &&) noexcept = default;


    //*************************************************************************************************************************************//
    //                                                             constructors                                                            //
    //*************************************************************************************************************************************//
    // TODO:

    explicit csvm(parameter<real_type> params = {}) : params_{ std::move(params) } {}

    template <typename... Args>
    explicit csvm(kernel_type kernel, Args&&... named_args);


    //*************************************************************************************************************************************//
    //                                                              fit model                                                              //
    //*************************************************************************************************************************************//
    template <typename label_type, typename... Args>
    [[nodiscard]] model<real_type, label_type> fit(const data_set<real_type, label_type> &data, Args&&... named_args);

    //*************************************************************************************************************************************//
    //                                                               predict                                                               //
    //*************************************************************************************************************************************//
    template <typename label_type = int>
    [[nodiscard]] std::vector<label_type> predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data);
    template <typename label_type = int>
    [[nodiscard]] std::vector<real_type> predict_values(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data);

    // calculate the accuracy of the model
    template <typename label_type = int>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model) {
        return this->score(model, model.data_);
    }
    // calculate the accuracy of the data_set
    template <typename label_type = int>
    [[nodiscard]] real_type score(const model<real_type, label_type> &model, const data_set<real_type> &data);

  protected:
    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Initialize the data on the respective device(s) (e.g. GPUs).
     */
    virtual void setup_data_on_device() = 0;
    /**
     * @brief Generate the vector `q`, a subvector of the least-squares matrix equation.
     * @return the generated `q` vector (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<real_type> generate_q(const std::vector<std::vector<real_type>> &data) = 0;
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] imax the maximum number of CG iterations
     * @param[in] eps error tolerance
     * @param[in] q subvector of the least-squares matrix equation
     * @return the alpha values
     */
    virtual std::vector<real_type> conjugate_gradient(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &b, const std::vector<real_type> &q, real_type QA_cost, real_type eps, size_type max_iter) = 0;
    /**
     * @brief Updates the normal vector #w_, used to speed-up the prediction in case of the linear kernel function, to the current data and alpha values.
     */
    virtual void update_w(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &alpha, size_type num_data_points, size_type num_features) = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @return a [`std::vector<real_type>`](https://en.cppreference.com/w/cpp/container/vector) filled with negative values for each prediction for a data point with the negative class and positive values otherwise (`[[nodiscard]]`)
     */
//    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) = 0;
    // TODO: API: sizes?!?
    [[nodiscard]] virtual std::vector<real_type> predict_values_impl(const std::vector<std::vector<real_type>> &support_vectors,
                                                                     const std::vector<real_type> &alpha,
                                                                     real_type rho,
                                                                     const std::vector<std::vector<real_type>> &predict_points) = 0;


    //*************************************************************************************************************************************//
    //                                              parameter initialized by the constructor                                               //
    //*************************************************************************************************************************************//

    parameter<real_type> params_{};

//    //*************************************************************************************************************************************//
//    //                                                         internal variables                                                          //
//    //*************************************************************************************************************************************//
//    /// The bias after learning.
//    real_type bias_{};
    /// The normal vector used for speeding up the prediction in case of the linear kernel function.
    std::vector<real_type> w_{};
};

/******************************************************************************
 *                                 constructor                                *
 ******************************************************************************/
template <typename T> template <typename... Args>
csvm<T>::csvm(kernel_type kernel, Args&&... named_args) {
    igor::parser p{ std::forward<Args>(named_args)... };

    // set kernel type
    params_.kernel = kernel;

    // compile time check: only named parameter are permitted
    static_assert(!p.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!p.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!p.has_other_than(gamma, degree, coef0, cost), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (p.has(gamma)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(gamma))>, decltype(params_.gamma)>, "gamma must be convertible to a real_type!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear) {
            std::clog << "gamma parameter provided to the linear kernel, which is not used!" << std::endl;
        }
        // set value
        params_.gamma = static_cast<decltype(params_.gamma)>(p(gamma));
    }
    if constexpr (p.has(degree)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(degree))>, decltype(params_.degree)>, "degree must be convertible to an int!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            std::clog << fmt::format("degree parameter provided to the {} kernel, which is not used!", kernel) << std::endl;
        }
        // set value
        params_.degree = static_cast<decltype(params_.degree)>(p(degree));
    }
    if constexpr (p.has(coef0)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(coef0))>, decltype(params_.coef0)>, "coef0 must be convertible to a real_type!");
        // runtime check: the value may only be used with a specific kernel type
        if (kernel == kernel_type::linear || kernel == kernel_type::rbf) {
            std::clog << fmt::format("coef0 parameter provided to the {} kernel, which is not used!", kernel) << std::endl;
        }
        // set value
        params_.coef0 = static_cast<decltype(params_.coef0)>(p(coef0));
    }
    if constexpr (p.has(cost)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(cost))>, decltype(params_.cost)>, "cost must be convertible to a real_type!");
        // set value
        params_.cost = static_cast<decltype(params_.cost)>(p(cost));
    }
}


/******************************************************************************
 *                                  fit model                                 *
 ******************************************************************************/
template <typename T> template <typename label_type, typename... Args>
auto csvm<T>::fit(const data_set<real_type, label_type> &data, Args&&... named_args) -> model<real_type, label_type> {
    igor::parser p{ std::forward<Args>(named_args)... };

    // set default values
    real_type epsilon_val = 0.001;
    size_type max_iter_val = data.num_features();

    // compile time check: only named parameter are permitted
    static_assert(!p.has_unnamed_arguments(), "Can only use named parameter!");
    // compile time check: each named parameter must only be passed once
    static_assert(!p.has_duplicates(), "Can only use each named parameter once!");
    // compile time check: only some named parameters are allowed
    static_assert(!p.has_other_than(epsilon, max_iter), "An illegal named parameter has been passed!");

    // compile time/runtime check: the values must have the correct types
    if constexpr (p.has(epsilon)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(epsilon))>, real_type>, "epsilon must be convertible to a real_type!");
        // set value
        epsilon_val = static_cast<real_type>(p(epsilon));
        // check if value makes sense
        if (epsilon_val <= real_type{ 0.0 }) {
            throw invalid_parameter_exception{ fmt::format("epsilon must be greater than 0, but is {}!", epsilon_val) };
        }
    }
    if constexpr (p.has(max_iter)) {
        // compile time check: the value must have the correct type
        static_assert(std::is_convertible_v<detail::remove_cvref_t<decltype(p(max_iter))>, size_type>, "max_iter must be convertible to a size_type!");
        // set value
        max_iter_val = static_cast<size_type>(p(max_iter));
        // check if value makes sense
        if (max_iter_val == size_type{ 0 }) {
            throw invalid_parameter_exception{ fmt::format("max_iter must be greater than 0, but is {}!", max_iter_val) };
        }
    }

    using namespace plssvm::operators;

    if (!data.has_labels()) {
        throw exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    }

    bool reset_gamma{ false };
    if (params_.gamma == real_type{ 0.0 }) {
        // no gamma provided -> use default value which depends on the number of features of the data set
        params_.gamma = real_type{ 1.0 } / data.num_features();
        reset_gamma = true;
    }

    // create model
    model<real_type, label_type> csvm_model{ params_, data };

    // move data to the device(s)
    this->setup_data_on_device();

    std::chrono::time_point start_time = std::chrono::steady_clock::now();

    real_type QA_cost{};
    std::vector<real_type> q{};
    std::vector<real_type> b = data.mapped_labels().value().get();
    const real_type b_back_value = b.back();
    #pragma omp parallel sections default(none) shared(q, data, b, b_back_value, QA_cost, params_)
    {
        #pragma omp section
        {
            q = generate_q(data.data());
        }
        #pragma omp section
        {
            b.pop_back();
            b -= b_back_value;
        }
        #pragma omp section
        {
            QA_cost = kernel_function(data.data().back(), data.data().back(), params_) + real_type{ 1.0 } / params_.cost;
        }
    }

    std::chrono::time_point end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Setup for solving the optimization problem done in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    start_time = std::chrono::steady_clock::now();

    // solve the minimization problem
    std::vector<real_type>& alpha = *csvm_model.alpha_ptr_;
    alpha = conjugate_gradient(data.data(), b, q, QA_cost, epsilon_val, max_iter_val);
    csvm_model.rho_ = -(b_back_value + QA_cost * sum(alpha) - (transposed{ q } * alpha));
    alpha.push_back(-sum(alpha));

    // TODO: necessary?
    w_.clear();

    // default gamma has been used -> reset gamma to 0.0
    if (reset_gamma) {
        params_.gamma = 0.0;
    }

    end_time = std::chrono::steady_clock::now();
    if (verbose) {
        fmt::print("Solved minimization problem (r = b - Ax) using CG in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    return csvm_model;
}


/******************************************************************************
 *                                   predict                                  *
 ******************************************************************************/
template <typename T> template <typename label_type>
auto csvm<T>::predict(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) -> std::vector<label_type> {
    if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict values
    const std::vector<real_type> predicted_values = predict_values(model, data);

    // convert predicted values to the correct labels
    std::vector<label_type> predicted_labels(predicted_values.size());
    #pragma omp parallel for default(none) shared(predicted_labels, predicted_values, model)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        predicted_labels[i] = model.data_.label_from_mapped_value(plssvm::operators::sign(predicted_values[i]));
    }
    return predicted_labels;
}

template <typename T> template <typename label_type>
auto csvm<T>::predict_values(const model<real_type, label_type> &model, const data_set<real_type, label_type> &data) -> std::vector<real_type> {
    if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // forward implementation to derived classes
    return predict_values_impl(model.data_.data(), *model.alpha_ptr_, model.rho_, data.data());
}

template <typename T> template <typename label_type>
auto csvm<T>::score(const model<real_type, label_type> &model, const data_set<real_type> &data) -> real_type {
    if (!data.has_labels()) {
        throw exception{ "the data set to score must have labels set!" };
    } else if (model.num_features() != data.num_features()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!", model.num_features(), data.num_features()) };
    }

    // predict labels
    const std::vector<label_type> predicted_labels = predict(model, data);
    // correct labels
    const std::vector<label_type>& correct_labels = model.labels().value();

    // calculate the accuracy
    size_type correct{ 0 };
    #pragma omp parallel for reduction(+ : correct) default(none) shared(predicted_labels, correct_labels)
    for (typename std::vector<label_type>::size_type i = 0; i < predicted_labels.size(); ++i) {
        if (predicted_labels[i] == correct_labels[i]) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(predicted_labels.size());
}

}  // namespace plssvm
