
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

add_executable(Preconditioner_tests matrix_view_test.cpp linalg_test.cpp preconditioners_test.cpp
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/matrix.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/shape.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/detail/string_utility.cpp>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/plssvm/exceptions/exceptions.cpp>
)
# ${PLSSVM_BASE_SOURCES})

set_property(TARGET Preconditioner_tests PROPERTY CXX_STANDARD 17)
add_sycl_to_target(TARGET Preconditioner_tests SOURCES matrix_view_test.cpp)
target_link_libraries(Preconditioner_tests PRIVATE GTest::gtest_main fmt::fmt)
target_include_directories(Preconditioner_tests PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
gtest_discover_tests(Preconditioner_tests)
