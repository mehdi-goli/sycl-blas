cmake_minimum_required(VERSION 2.8.2)
project(syclblas_benchmark)

add_executable(syclblas_benchmarks syclblas_benchmark.cpp)
set_property(TARGET syclblas_benchmarks PROPERTY CXX_STANDARD 11)
add_sycl_to_target(syclblas_benchmarks ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/syclblas_benchmark.cpp)

find_package(PkgConfig)

#find_package(CLBlast REQUIRED)
pkg_search_module(CLBLAST REQUIRED clblast)
pkg_check_modules(CLBLAST clblast REQUIRED)
add_executable(clblast_benchmarks clblast_benchmark.cpp)
set_property(TARGET clblast_benchmarks PROPERTY CXX_STANDARD 11)
message("clblast ${CLBLAST_LIBRARIES}")
include_directories(${CLBLAST_INCLUDE_DIRS})
target_link_libraries(clblast_benchmarks PUBLIC ${CLBLAST_LIBRARIES})
target_link_libraries(clblast_benchmarks PUBLIC ${OpenCL_LIBRARIES})
