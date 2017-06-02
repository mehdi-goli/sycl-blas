#pragma once

#include <cstdlib>
#include <complex>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <interface/blas1_interface_sycl.hpp>

using namespace cl::sycl;
using namespace blas;

// T = type, C = container, E = executor

// wrapper for the arguments
template <class T, template <class... As> class C, class E>
struct blas_templ_struct {
  using type = T;
  template <class U = T>
    using container = C<U>;
  using executor = E;
};
template <class T, template <class... As> class C = std::vector, class E = SYCL>
using blas_args = blas_templ_struct<T, C, E>;

//
template <class B> class BLAS1_Test;

template <class T, template <class... As> class C, class E>
class BLAS1_Test <blas_args<T, C, E>> : public ::testing::Test {
public:
  BLAS1_Test(){}

  virtual ~BLAS1_Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <class U = T>
  static void set_rand(C<U> &vec, std::pair <U, U> &&bounds) {
    for(auto &x : vec) {
      x = U(rand()) % (bounds.right - bounds.left) - bounds.right;
    }
  }

  template <class U = T>
  static C<U> make_randcont(size_t size, std::pair<U, U> &&bounds) {
    C<U> container(size);
    set_rand(container);
    return container;
  }

  template <class U = T>
  static buffer<U, 1> make_buffer(C <U> &vec) {
    return buffer<U, 1>(vec.data(), vec.size());
  }

  template <class U = T>
  static vector_view<U, buffer<U>> make_vview(buffer<U, 1> &buf) {
    return vector_view<U, buffer<U>>(buf);
  }

  static bool cmp_values(T a, T b, T prec = 1e-6) {
    return std::abs(a - b) < prec;
  }

  template <typename = typename std::enable_if<std::is_same<E, SYCL>::value>::type>
  static cl::sycl::queue make_queue() {
    return cl::sycl::queue([=](cl::sycl::exception_list eL) {
      try {
        for (auto &e : eL)
          std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });
  }
};
