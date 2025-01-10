# Hacked together unit tests for most of the preconditioner functionality.
# Pretty much all tests require the shared allocation mode.

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
        DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
enable_testing()

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(fmt 10.2.1 QUIET)

add_executable(preconditioner_tests
        matrix/common.cpp matrix/utility.cpp
        linalg/arithmetic.cpp linalg/svd.cpp linalg/cholesky.cpp
        preconditioners/jacobi.cpp preconditioners/rpcholesky.cpp
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/matrix.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/shape.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/detail/string_utility.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/exceptions/exceptions.cpp>
)

set_property(TARGET preconditioner_tests PROPERTY CXX_STANDARD 17)
# enable fast math
target_compile_options(preconditioner_tests PRIVATE $<$<CONFIG:Release>:-ffast-math>)
add_sycl_to_target(TARGET preconditioner_tests SOURCES matrix_view.cpp)
target_link_libraries(preconditioner_tests PRIVATE GTest::gtest_main fmt::fmt)

# This is an unfortunate workaround to avoid having to include all of the dependencies that plssvm includes,
# e.g. igor etc., some code uses this define to guard includes of those headers if necessary.
# Example: including plssvm/detail/logging includes igor internally so we guard against logging in unit tests
# with this define.
target_compile_definitions(preconditioner_tests PRIVATE RUNNING_GTEST)


target_include_directories(preconditioner_tests PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
target_include_directories(preconditioner_tests PUBLIC
        "/home/jns/dev/thesis/external/eigen")
gtest_discover_tests(preconditioner_tests)
