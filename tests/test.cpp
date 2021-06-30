// #include "CSVM.hpp"
#include "mocks/CSVM.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

#include "plssvm/exceptions.hpp"

#include "plssvm/OpenMP/OpenMP_CSVM.hpp"

TEST(IO, libsvmFormat) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.libsvmParser(TESTPATH "/data/5x4.libsvm"); //TODO: add comments etc to libsvm test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        {-1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288},
        {-0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026},
        {0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387},
        {-0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827},
        {1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514},
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, sparselibsvmFormat) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.libsvmParser(TESTPATH "/data/5x4.sparse.libsvm"); //TODO: add comments etc to libsvm test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        {0., 0., 0., 1.0978832703949288},
        {0., 0., 0.51687296029754564, 0.},
        {0., 1.01405596624706053, 0., 0.},
        {0., 0.60276937379453293, -0.13086851759108944, 0.},
        {0., 0., 0.298499933047586044, 0.},
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, arffFormat) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.arffParser(TESTPATH "/data/5x4.arff"); //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        {-1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288},
        {-0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026},
        {0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387},
        {-0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827},
        {1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514},
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, arffParserGamma) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.arffParser(TESTPATH "/data/5x4.arff"); //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(1., 1., 0, 1., 0, 1., false);
    csvm_gammazero.arffParser(TESTPATH "/data/5x4.arff"); //TODO: add comments etc to arff test file
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(IO, libsvmParserGamma) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.libsvmParser(TESTPATH "/data/5x4.libsvm"); //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(1., 1., 0, 1., 0, 1., false);
    csvm_gammazero.libsvmParser(TESTPATH "/data/5x4.libsvm"); //TODO: add comments etc to arff test file
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(IO, writeModel) {
    MockCSVM csvm(1., 0.001, 0, 3.0, 0.0, 0.0, false);
    csvm.libsvmParser(TESTPATH "/data/5x4.libsvm"); //TODO: add comments etc to arff test file
    std::string model1 = std::tmpnam(nullptr);
    csvm.writeModel(model1);
    std::ifstream model1_ifs(model1);
    std::string genfile1((std::istreambuf_iterator<char>(model1_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model1.c_str());

    EXPECT_THAT(genfile1, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv 0+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    std::string model2 = std::tmpnam(nullptr); // TODO: only if openmp backend is available
    MockOpenMP_CSVM csvm2(1., 0.001, 0, 3.0, 0.0, 0.0, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model2);

    std::ifstream model2_ifs(model2);
    std::string genfile2((std::istreambuf_iterator<char>(model2_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model2.c_str());

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*\?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
#else
#pragma message("Ignore OpenMP backend test")
#endif
}

TEST(IO, libsvmFormatIllFormed) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_THROW(csvm.libsvmParser(TESTPATH "/data/5x4.arff");, plssvm::invalid_file_format_exception);
}

TEST(IO, arffFormatIllFormed) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_THROW(csvm.arffParser(TESTPATH "/data/5x4.libsvm");, plssvm::invalid_file_format_exception);
}

TEST(IO, libsvmNoneExistingFile) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_THROW(csvm.libsvmParser(TESTPATH "/data/5x5.ar");, plssvm::file_not_found_exception);
}

TEST(IO, arffNoneExistingFile) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_THROW(csvm.arffParser(TESTPATH "/data/5x5.lib");, plssvm::file_not_found_exception);
}

TEST(kernel, linear) {
    const real_t degree = 0.0;
    const real_t gamma = 0.0;
    const real_t coef0 = 0.0;
    const size_t size = 512;
    std::vector<real_t> x1(size);
    std::vector<real_t> x2(size);
    real_t correct = 0;
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    for (size_t i = 0; i < size; ++i) {
        correct += x1[i] * x2[i];
    }

    MockCSVM csvm(1., 0.001, 0, degree, gamma, coef0, false);
    real_t result = csvm.kernel_function(x1, x2);
    real_t result2 = csvm.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result);
    EXPECT_DOUBLE_EQ(correct, result2);

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    MockOpenMP_CSVM csvm_OpenMP(1., 0.001, 0, degree, gamma, coef0, false);
    real_t result_OpenMP = csvm_OpenMP.kernel_function(x1, x2);
    real_t result2_OpenMP = csvm_OpenMP.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenMP);
    EXPECT_DOUBLE_EQ(correct, result2_OpenMP);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    MockOpenCL_CSVM csvm_OpenCL(1., 0.001, 0, degree, gamma, coef0, false);
    real_t result_OpenCL = csvm_OpenCL.kernel_function(x1, x2);
    real_t result2_OpenCL = csvm_OpenCL.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenCL);
    EXPECT_DOUBLE_EQ(correct, result2_OpenCL);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    MockCUDA_CSVM csvm_CUDA(1., 0.001, 0, degree, gamma, coef0, false);
    real_t result_CUDA = csvm_CUDA.kernel_function(x1, x2);
    real_t result2_CUDA = csvm_CUDA.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_CUDA);
    EXPECT_DOUBLE_EQ(correct, result2_CUDA);
#endif
}