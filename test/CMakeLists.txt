cmake_minimum_required(VERSION 3.0.2)

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
file(GLOB SYCL_SRCS
  ${SYCLBLAS_TEST}/blas1_simple_axpy.cpp
  ${SYCLBLAS_TEST}/blas1_rand_copy_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_swap_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_axpy_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_asum_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_dot_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_nrm2_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_imax_test.cpp
  ${SYCLBLAS_TEST}/blas1_rand_imin_test.cpp
)

foreach(blas_test ${SYCL_SRCS})
  message("-- blas test: ${blas_test}")
  get_filename_component(test_exec ${blas_test} NAME_WE)
  message("-- test exec: ${test_exec}")
  set(test_exec, ${blas_test})
  add_executable(${test_exec} main.cpp ${blas_test})
  set_property(TARGET ${test_exec} PROPERTY CXX_STANDARD 11)
  target_link_libraries(${test_exec} PUBLIC libgtest libgmock)
  add_sycl_to_target(${test_exec} ${CMAKE_CURRENT_BINARY_DIR} ${blas_test})
  add_test(name ${test_exec} command ${CMAKE_CURRENT_BINARY_DIR}/${test_exec})
  message("-- Created google test ${test_exec}")
endforeach()
