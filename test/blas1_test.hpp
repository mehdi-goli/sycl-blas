#ifndef BLAS1_TEST_HPP_DFQO1OHP
#define BLAS1_TEST_HPP_DFQO1OHP

#include <complex>
#include <cstdlib>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
template <class B>
class BLAS1_Test;

template <class T, template <class... As> class C, class E>
class BLAS1_Test<blas_args<T, C, E>> : public ::testing::Test {
public:
  BLAS1_Test() {}

  virtual ~BLAS1_Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static size_t rand_size() {
    return rand() % size_t(1.5e7);
  }

  template <class U = T>
  static void set_rand(C<U> &vec, std::pair<U, U> &bounds) {
    for (auto &x : vec) {
      x = U(rand() % int(bounds.first - bounds.second)) - bounds.second;
    }
  }

  template <class U = T>
  static C<U> make_randcont(size_t size, std::pair<U, U> &&bounds = {-100, 100}) {
    C<U> container(size);
    set_rand(container, bounds);
    return container;
  }

  template <class U = T>
  static void print_cont(const C<U> &vec, std::string name = "vector") {
    std::cout << name << ": ";
    for(auto &e : vec)
      std::cout << e << " ";
    std::cout << std::endl;
  }

  template <class U = T>
  static buffer<U, 1> make_buffer(C<U> &vec) {
    return buffer<U, 1>(vec.data(), vec.size());
  }

  template <class U = T>
  static vector_view<U, buffer<U>> make_vview(buffer<U, 1> &buf) {
    return vector_view<U, buffer<U>>(buf);
  }

  static bool eq_vals(T a, T b, T prec = 1e-6) {
    bool ret = std::abs(a - b) < prec;
    if(!ret) {
      std::cout << "not equal: " << a << " vs " << b << std::endl;
    }
    return ret;
  }

  template <
      typename = typename std::enable_if<std::is_same<E, SYCL>::value>::type>
  static cl::sycl::queue make_queue() {
    return cl::sycl::queue([=](cl::sycl::exception_list eL) {
      try {
        for (auto &e : eL) std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << " E " << e.what() << std::endl;
      } catch (...) {
        std::cout << " An exception " << std::endl;
      }
    });
  }
};

// wrapping for google test

template <class B, class T = typename B::type>
using Container = typename B::template container<T>;

template <class B> using TEST_B = BLAS1_Test<B>;

typedef ::testing::Types<blas_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

// unpacking the parameters within the test function
#define B1_TEST(name) TYPED_TEST(BLAS1_Test, name)
#define UNPACK_PARAM \
  using B=TypeParam; \
  using T=typename B::type; \
  using _T=TEST_B<B>; \
  using E=typename B::executor;
#define EXECUTE(name) auto q = _T::make_queue(); Executor<E> name(q);
#define TO_VIEW(name) \
  auto buf_##name = _T::make_buffer(name); \
  auto view_##name = _T::make_vview(buf_##name);

#endif
