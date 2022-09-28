/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the data_set used for learning an SVM model.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::{convert_to, split_as}
#include "plssvm/exceptions/exceptions.hpp"
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "utility.hpp"  // util::create_temp_file, util::redirect_output, EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // EXPECT_EQ, EXPECT_TRUE, ASSERT_GT, GTEST_FAIL, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                          // ::testing::{Types, StaticAssertTypeEq, Test, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <regex>        // std::regex, std::regex_match, std::regex::extended
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

// struct for the used type combinations
template <typename T, typename U>
struct type_combinations {
    using real_type = T;
    using label_type = U;
};

// the floating point and label types combinations to test
using type_combinations_types = ::testing::Types<
    type_combinations<float, int>,
    type_combinations<float, std::string>,
    type_combinations<double, int>,
    type_combinations<double, std::string>>;

////////////////////////////////////////////////////////////////////////////////
////                          scaling nested-class                          ////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DataSetScaling : public ::testing::Test, private util::redirect_output {};
TYPED_TEST_SUITE(DataSetScaling, type_combinations_types);

TYPED_TEST(DataSetScaling, default_construct_factor) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    factor_type factor{};

    // test values
    EXPECT_EQ(factor.feature, std::size_t{});
    EXPECT_EQ(factor.lower, real_type{});
    EXPECT_EQ(factor.upper, real_type{});
}
TYPED_TEST(DataSetScaling, construct_factor) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    factor_type factor{ 1, real_type{ -2.5 }, real_type{ 2.5 } };

    // test values
    EXPECT_EQ(factor.feature, 1);
    EXPECT_EQ(factor.lower, -2.5);
    EXPECT_EQ(factor.upper, 2.5);
}

TYPED_TEST(DataSetScaling, construct_interval) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ real_type{ -1.0 }, real_type{ 1.0 } };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, real_type{ -1.0 });
    EXPECT_EQ(scale.scaling_interval.second, real_type{ 1.0 });
    EXPECT_TRUE(scale.scaling_factors.empty());
}
TYPED_TEST(DataSetScaling, construct_invalid_interval) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class with an invalid interval
    EXPECT_THROW_WHAT((scaling_type{ real_type{ 1.0 }, real_type{ -1.0 } }), plssvm::data_set_exception, "Inconsistent scaling interval specification: lower (1) must be less than upper (-1)!");
}
TYPED_TEST(DataSetScaling, construct_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, plssvm::detail::convert_to<real_type>("-1.4"));
    EXPECT_EQ(scale.scaling_interval.second, plssvm::detail::convert_to<real_type>("2.6"));
    std::vector<factors_type> correct_factors = {
        factors_type{ 0, real_type{ 0.0 }, real_type{ 1.0 } },
        factors_type{ 1, real_type{ 1.1 }, real_type{ 2.1 } },
        factors_type{ 3, real_type{ 3.3 }, real_type{ 4.3 } },
        factors_type{ 4, real_type{ 4.4 }, real_type{ 5.4 } },
    };
    ASSERT_EQ(scale.scaling_factors.size(), correct_factors.size());
    for (std::size_t i = 0; i < correct_factors.size(); ++i) {
        EXPECT_EQ(scale.scaling_factors[i].feature, correct_factors[i].feature);
        EXPECT_EQ(scale.scaling_factors[i].lower, correct_factors[i].lower);
        EXPECT_EQ(scale.scaling_factors[i].upper, correct_factors[i].upper);
    }
}

TYPED_TEST(DataSetScaling, save) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create temporary file
    const std::string filename = util::create_temp_file();
    {
        // save scaling factors
        scale.save(filename);

        // read file and check its content
        plssvm::detail::io::file_reader reader{ filename };
        reader.read_lines('#');

        std::vector<std::string> regex_patterns;
        // header
        regex_patterns.emplace_back("x");
        regex_patterns.emplace_back("[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
        // the remaining values
        regex_patterns.emplace_back("\\+?[1-9]+[0-9]* [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");

        // check the content
        ASSERT_GE(reader.num_lines(), 2);
        std::regex reg{ regex_patterns[0], std::regex::extended };  // check the line only containing an x
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(0) }, reg));
        reg = std::regex{ regex_patterns[1], std::regex::extended };  // check the scaling interval line
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
        reg = std::regex{ regex_patterns[2], std::regex::extended };  // check the remaining lines
        for (std::size_t i = 2; i < reader.num_lines(); ++i) {
            EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
        }
    }
    std::filesystem::remove(filename);
}
TYPED_TEST(DataSetScaling, save_empty_scaling_factors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ real_type{ -1.0 }, real_type{ 1.0 } };

    // create temporary file
    const std::string filename = util::create_temp_file();
    {
        // save scaling factors
        scale.save(filename);

        // read file and check its content
        plssvm::detail::io::file_reader reader{ filename };
        reader.read_lines('#');

        std::vector<std::string> regex_patterns;
        // header
        regex_patterns.emplace_back("x");
        regex_patterns.emplace_back("[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");

        // check the content
        ASSERT_EQ(reader.num_lines(), 2);
        std::regex reg{ regex_patterns[0], std::regex::extended };  // check the line only containing an x
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(0) }, reg));
        reg = std::regex{ regex_patterns[1], std::regex::extended };  // check the scaling interval line
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
    }
    std::filesystem::remove(filename);
}

////////////////////////////////////////////////////////////////////////////////
////                       label mapper nested-class                        ////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DataSetLabelMapper : public ::testing::Test {};
TYPED_TEST_SUITE(DataSetLabelMapper, type_combinations_types);

TYPED_TEST(DataSetLabelMapper, construct) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        different_labels = std::vector<int>{ -64, 32 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        different_labels = std::vector<std::string>{ "cat", "dog" };
    }

    // create label mapper
    const std::vector<label_type> labels = {
        different_labels[0],
        different_labels[1],
        different_labels[1],
        different_labels[0],
        different_labels[1]
    };
    const label_mapper_type mapper{ labels };

    // test values
    EXPECT_EQ(mapper.num_mappings(), 2);
    EXPECT_EQ(mapper.labels(), different_labels);
    // test mapping
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ -1 }), different_labels[0]);
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ 1 }), different_labels[1]);
    EXPECT_EQ(mapper.get_mapped_value_by_label(different_labels[0]), real_type{ -1 });
    EXPECT_EQ(mapper.get_mapped_value_by_label(different_labels[1]), real_type{ 1 });
}
TYPED_TEST(DataSetLabelMapper, construct_too_many_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        different_labels = std::vector<int>{ -64, 32, 128 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        different_labels = std::vector<std::string>{ "cat", "dog", "mouse" };
    }

    // too many labels provided!
    EXPECT_THROW_WHAT(label_mapper_type{ different_labels }, plssvm::data_set_exception, "Currently only binary classification is supported, but 3 different labels were given!");
}
TYPED_TEST(DataSetLabelMapper, get_mapped_value_by_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("-10")), real_type{ -1 });
    EXPECT_EQ(mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("10")), real_type{ 1 });
}
TYPED_TEST(DataSetLabelMapper, get_mapped_value_by_invalid_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("42")), plssvm::data_set_exception, "Label \"42\" unknown in this label mapping!");
}
TYPED_TEST(DataSetLabelMapper, get_label_by_mapped_value) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ -1 }), plssvm::detail::convert_to<label_type>("-10"));
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ 1 }), plssvm::detail::convert_to<label_type>("10"));
}
TYPED_TEST(DataSetLabelMapper, get_label_by_invalid_mapped_value) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_label_by_mapped_value(real_type{ 0.0 }), plssvm::data_set_exception, "Mapped value \"0\" unknown in this label mapping!");
}
TYPED_TEST(DataSetLabelMapper, num_mappings) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-1"), plssvm::detail::convert_to<label_type>("1") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.num_mappings(), 2);
}
TYPED_TEST(DataSetLabelMapper, labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-1"), plssvm::detail::convert_to<label_type>("1") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.labels(), different_labels);
}

////////////////////////////////////////////////////////////////////////////////
////                             data_set class                             ////
////////////////////////////////////////////////////////////////////////////////

// the floating point and label types combinations to test
using extended_type_combinations_types = ::testing::Types<
    type_combinations<float, int>,
    type_combinations<float, std::string>,
    type_combinations<float, bool>,
    type_combinations<double, int>,
    type_combinations<double, std::string>,
    type_combinations<double, bool>>;

template <typename T>
const std::vector<std::vector<T>> correct_data_points_arff = {
    plssvm::detail::split_as<T>("-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288", ','),
    plssvm::detail::split_as<T>("-0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026", ','),
    plssvm::detail::split_as<T>("0.0,0.60276937379453293,-0.13086851759108944,0.0", ','),
    plssvm::detail::split_as<T>("0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387", ','),
    plssvm::detail::split_as<T>("1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514", ',')
};
template <typename T>
const std::vector<std::vector<T>> correct_data_points_libsvm = {
    plssvm::detail::split_as<T>("-1.117827500607882 -2.9087188881250993 0.66638344270039144 1.0978832703949288", ' '),
    plssvm::detail::split_as<T>("-0.5282118298909262 -0.335880984968183973 0.51687296029754564 0.54604461446026", ' '),
    plssvm::detail::split_as<T>("0.57650218263054642 1.01405596624706053 0.13009428079760464 0.7261913886869387", ' '),
    plssvm::detail::split_as<T>("-0.20981208921241892 0.60276937379453293 -0.13086851759108944 0.10805254527169827", ' '),
    plssvm::detail::split_as<T>("1.88494043717792 1.00518564317278263 0.298499933047586044 1.6464627048813514", ' ')
};
template <typename T>
std::pair<std::vector<std::vector<T>>, std::vector<std::tuple<std::size_t, T, T>>> scale(const std::vector<std::vector<T>> &data, const T lower, const T upper) {
    std::vector<std::tuple<std::size_t, T, T>> factors(data.front().size(), std::make_tuple(0, std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest()));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::get<0>(factors[i]) = i;
        for (std::size_t j = 0; j < data.size(); ++j) {
            std::get<1>(factors[i]) = std::min(std::get<1>(factors[i]), data[j][i]);
            std::get<2>(factors[i]) = std::max(std::get<2>(factors[i]), data[j][i]);
        }
    }
    std::vector<std::vector<T>> ret = data;
    for (std::size_t i = 0; i < ret.size(); ++i) {
        for (std::size_t j = 0; j < ret.front().size(); ++j) {
            ret[i][j] = lower + (upper - lower) * (data[i][j] - std::get<1>(factors[j])) / (std::get<2>(factors[j]) - std::get<1>(factors[j]));
        }
    }
    return std::make_pair(std::move(ret), std::move(factors));
}
template <typename T>
std::vector<T> correct_labels() {
    if constexpr (std::is_same_v<T, int>) {
        return std::vector<int>{ 1, 1, -1, -1, -1 };
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::vector<std::string>{ "cat", "cat", "dog", "dog", "dog" };
    }
}
template <typename T>
const std::vector<T> correct_mapped_values = { T{ 1.0 }, T{ 1.0 }, T{ -1.0 }, T{ -1.0 }, T{ -1.0 } };
template <typename T>
std::vector<T> correct_different_labels() {
    if constexpr (std::is_same_v<T, int>) {
        return std::vector<int>{ -1, 1 };
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::vector<std::string>{ "cat", "dog" };
    }
}

template <typename T>
class DataSet : public ::testing::Test, private util::redirect_output {};
TYPED_TEST_SUITE(DataSet, type_combinations_types);

TYPED_TEST(DataSet, construct_arff_from_file_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/arff/5x4_{}.arff", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_arff<real_type>);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_arff<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_arff<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_arff_from_file_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/arff/5x4_{}_without_label.arff", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_arff<real_type>);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.different_labels().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_arff<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_arff<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_libsvm_from_file_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_libsvm<real_type>);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_libsvm<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_libsvm<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_libsvm_from_file_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}_without_label.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_libsvm<real_type>);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.different_labels().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_libsvm<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_libsvm<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_explicit_arff_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/arff/5x4_{}.arff", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, plssvm::file_format_type::arff };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_arff<real_type>);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_arff<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_arff<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_explicit_libsvm_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, plssvm::file_format_type::libsvm };

    // check values
    EXPECT_EQ(data.data(), correct_data_points_libsvm<real_type>);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_libsvm<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_libsvm<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSet, construct_scaled_arff_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/arff/5x4_{}.arff", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, { real_type{ -1.0 }, real_type{ 1.0 } } };

    // check values
    const auto [scaled_data_points, scaling_factors] = scale(correct_data_points_arff<real_type>, real_type{ -1.0 }, real_type{ 1.0 });
    EXPECT_EQ(data.data(), scaled_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_arff<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_arff<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_libsvm_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, { real_type{ -2.5 }, real_type{ 2.5 } } };

    // check values
    const auto [scaled_data_points, scaling_factors] = scale(correct_data_points_libsvm<real_type>, real_type{ -2.5 }, real_type{ 2.5 });
    EXPECT_EQ(data.data(), scaled_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_libsvm<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_libsvm<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSet, construct_scaled_explicit_arff_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/arff/5x4_{}.arff", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, plssvm::file_format_type::arff, { real_type{ -1.0 }, real_type{ 1.0 } } };

    // check values
    const auto [scaled_data_points, scaling_factors] = scale(correct_data_points_arff<real_type>, real_type{ -1.0 }, real_type{ 1.0 });
    EXPECT_EQ(data.data(), scaled_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_arff<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_arff<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_explicit_libsvm_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    plssvm::data_set<real_type, label_type> data{ filename, plssvm::file_format_type::libsvm, { real_type{ -2.5 }, real_type{ 2.5 } } };

    // check values
    const auto [scaled_data_points, scaling_factors] = scale(correct_data_points_libsvm<real_type>, real_type{ -2.5 }, real_type{ 2.5 });
    EXPECT_EQ(data.data(), scaled_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels<label_type>());
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), correct_different_labels<label_type>());

    EXPECT_EQ(data.num_data_points(), correct_data_points_libsvm<real_type>.size());
    EXPECT_EQ(data.num_features(), correct_data_points_libsvm<real_type>.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSet, scale_too_many_factors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    // create (invalid) scaling factors
    scaling_type scaling{ real_type{ -1.0 }, real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 0, 0.0, 0.1 },
        factors_type{ 1, 1.0, 1.1 },
        factors_type{ 2, 2.0, 2.1 },
        factors_type{ 3, 3.0, 3.1 },
        factors_type{ 4, 4.0, 4.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ filename, scaling }), plssvm::data_set_exception, "Need at most as much scaling factors as features in the data set are present (4), but 5 were given!");
}
TYPED_TEST(DataSet, scale_invalid_feature_index) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    // create (invalid) scaling factors
    scaling_type scaling{ real_type{ -1.0 }, real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 4, 4.0, 4.1 },
        factors_type{ 2, 2.0, 2.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ filename, scaling }), plssvm::data_set_exception, "The maximum scaling feature index most not be greater than 3, but is 4!");
}
TYPED_TEST(DataSet, scale_duplicate_feature_index) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    const std::string filename = fmt::format("{}/data/libsvm/5x4_{}.libsvm", PLSSVM_TEST_PATH, std::is_same_v<label_type, int> ? "int" : "string");
    // create (invalid) scaling factors
    scaling_type scaling{ real_type{ -1.0 }, real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 1, 1.0, 1.1 },
        factors_type{ 2, 2.0, 2.1 },
        factors_type{ 3, 3.0, 3.1 },
        factors_type{ 2, 2.0, 2.1 }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ filename, scaling }), plssvm::data_set_exception, "Found more than one scaling factor for the feature index 2!");
}

TYPED_TEST(DataSet, construct_from_vector_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.0 }, real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.0 }, real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };

    // create data set
    const plssvm::data_set<real_type, label_type> data{ correct_data_points };

    // check values
    EXPECT_EQ(data.data(), correct_data_points);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.different_labels().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points.size());
    EXPECT_EQ(data.num_features(), correct_data_points.front().size());
    EXPECT_EQ(data.num_different_labels(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_from_empty_vector) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points
    const std::vector<std::vector<real_type>> correct_data_points;

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ correct_data_points }), plssvm::data_set_exception, "Data vector is empty!");
}
TYPED_TEST(DataSet, construct_from_vector_with_differing_num_features) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 } }
    };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ correct_data_points }), plssvm::data_set_exception, "All points in the data vector must have the same number of features!");
}
TYPED_TEST(DataSet, construct_from_vector_with_no_features) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points
    const std::vector<std::vector<real_type>> correct_data_points = { {}, {} };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ correct_data_points }), plssvm::data_set_exception, "No features provided for the data points!");
}

TYPED_TEST(DataSet, construct_from_vector_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points and labels
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.0 }, real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.0 }, real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> labels;
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        labels = std::vector<label_type>{ -1, 1, -1, 1 };
        different_labels = std::vector<label_type>{ -1, 1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        labels = std::vector<label_type>{ "cat", "dog", "cat", "dog" };
        different_labels = std::vector<label_type>{ "cat", "dog" };
    }

    // create data set
    const plssvm::data_set<real_type, label_type> data{ correct_data_points, labels };

    // check values
    EXPECT_EQ(data.data(), correct_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.size());
    EXPECT_EQ(data.num_features(), correct_data_points.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSet, construct_from_vector_mismatching_num_data_points_and_labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points and labels
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.0 }, real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.0 }, real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> labels;
    if constexpr (std::is_same_v<label_type, int>) {
        labels = std::vector<label_type>{ -1, 1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        labels = std::vector<label_type>{ "cat", "dog", "cat" };
    }

    // create data set
    EXPECT_THROW_WHAT((plssvm::data_set<real_type, label_type>{ correct_data_points, labels }), plssvm::data_set_exception, "Number of labels (3) must match the number of data points (4)!");
}

TYPED_TEST(DataSet, construct_scaled_from_vector_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create data points
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.0 }, real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.0 }, real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };

    // create data set
    const plssvm::data_set<real_type, label_type> data{ correct_data_points, scaling_type{ real_type{ -1.0 }, real_type{ 1.0 } } };

    const auto [correct_data_points_scaled, scaling_factors] = scale(correct_data_points, real_type{ -1.0 }, real_type{ 1.0 });
    // check values
    EXPECT_EQ(data.data(), correct_data_points_scaled);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.different_labels().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.size());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.front().size());
    EXPECT_EQ(data.num_different_labels(), 0);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSet, construct_scaled_from_vector_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create data points and labels
    const std::vector<std::vector<real_type>> correct_data_points = {
        { real_type{ 0.0 }, real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 } },
        { real_type{ 1.0 }, real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.0 }, real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.0 }, real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> labels;
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        labels = std::vector<label_type>{ -1, 1, -1, 1 };
        different_labels = std::vector<label_type>{ -1, 1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        labels = std::vector<label_type>{ "cat", "dog", "cat", "dog" };
        different_labels = std::vector<label_type>{ "cat", "dog" };
    }

    // create data set
    const plssvm::data_set<real_type, label_type> data{ correct_data_points, labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = scale(correct_data_points, real_type{ -1.0 }, real_type{ 1.0 });
    // check values
    EXPECT_EQ(data.data(), correct_data_points_scaled);
    EXPECT_TRUE(data.has_labels());
    EXPECT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    EXPECT_TRUE(data.different_labels().has_value());
    EXPECT_EQ(data.different_labels().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.size());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.front().size());
    EXPECT_EQ(data.num_different_labels(), 2);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.lower, std::get<1>(scaling_factors[i]));
        util::gtest_assert_floating_point_near(factors.upper, std::get<2>(scaling_factors[i]));
    }
}