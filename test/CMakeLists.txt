cmake_minimum_required(VERSION 2.8.2)

project(syclblas_gtest)

include(ExternalProject)

# compile google test, google mock

set(SYCLBLAS_TEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

ExternalProject_Add(
	gtest
	URL https://github.com/google/googletest/archive/release-1.8.0.zip
	PREFIX ${SYCLBLAS_TEST_INCLUDE}/gtest
	INSTALL_COMMAND ""
)

add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

ExternalProject_Get_Property(gtest source_dir binary_dir)
set_target_properties(libgtest PROPERTIES
	IMPORTED_LOCATION "${binary_dir}/googlemock/gtest/libgtest.a"
	IMPORTED_LINK_INTERFACE_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}"
)
set(GTEST_LIB "${source_dir}/googletest/include")

add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gmock)

set_target_properties(libgmock PROPERTIES
	"IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
	"IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)
set(GMOCK_LIB "${source_dir}/googlemock/include")

include_directories(
  ${GTEST_LIB}
  ${GMOCK_LIB}
)

# compiling tests
file(GLOB SRCS *.cpp)
add_executable(blas1_test ${SRCS})
set_property(TARGET blas1_test PROPERTY CXX_STANDARD 11)
target_link_libraries(blas1_test PUBLIC libgtest libgmock)
add_sycl_to_target(blas1_test ${CMAKE_CURRENT_BINARY_DIR} ${SYCLBLAS_TEST}/blas1_simple_axpy.cpp)

enable_testing()

add_test(name blas1_test command ${CMAKE_CURRENT_BINARY_DIR}/blas1_test)
message("-- Created google tests")
