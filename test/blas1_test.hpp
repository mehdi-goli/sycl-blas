#ifndef BLAS1_TEST_HPP_DFQO1OHP
#define BLAS1_TEST_HPP_DFQO1OHP

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <interface/blas1_interface_sycl.hpp>

using namespace cl::sycl;
using namespace blas;

// sfinae for sample size
template <typename C> struct option_size;
#define RANDOM_SIZE UINT_MAX
#define REGISTER_SIZE(size, test_class) template <> struct option_size<class test_class> { static constexpr const size_t value = size; };

// sfinae for precision registered for a type
template <class T, typename C> struct option_prec;
#define REGISTER_PREC(type, val, Test_Class) template <> struct option_prec<type, class Test_Class> { static constexpr const type value = val; };

// T = type, C = container, E = executor

// wrapper for the arguments
template <class T, template <class... As> class C, class E>
struct blas_templ_struct {
  using type = T;
  template <class U = T> using container = C<U>;
  using executor = E;
};
template <class T, template <class... As> class C = std::vector, class E = SYCL>
using blas_args = blas_templ_struct<T, C, E>;

//
template <class B>
class BLAS1_Test;

template <class T_, template <class... As> class C_, class E_>
class BLAS1_Test<blas_args<T_, C_, E_>> : public ::testing::Test {
 public:
  using T = T_;
  template <class U = T> using C = C_<U>;
  using E = E_;

  BLAS1_Test() {}

  virtual ~BLAS1_Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <class U = T>
  static size_t rand_size() {
    size_t ret = rand() >> 5;
    int type_size = sizeof(U) * CHAR_BIT - std::numeric_limits<U>::digits10 - 2;
    return (ret & (std::numeric_limits<size_t>::max() + (size_t(1) << (type_size - 2)))) + 1;
  }

  template <class U = T>
  static void set_rand(C<U> &vec, std::pair<U, U> &bounds) {
    for (auto &x : vec) {
      x = U(rand() % int(bounds.first - bounds.second) * 1000) * .001 -
          bounds.second;
    }
  }

  template <class U = T>
  static C<U> make_randcont(size_t size, std::pair<U, U> &&bounds = {-1, 1}) {
    C<U> container(size);
    set_rand(container, bounds);
    return container;
  }

  template <class U = T>
  static void print_cont(const C<U> &vec, std::string name = "vector") {
    std::cout << name << ": ";
    for (auto &e : vec) std::cout << e << " ";
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

// it is important that all tests are run with the same test size
template <class _T>
size_t test_size() {
  /* return 1e6; */
  using T = typename _T::T;
  static bool first = true;
  static size_t N;
  if (first) {
    first = false;
    N = _T::rand_size();
  }
  return N;
}

// wrapping for google test

template <class _T, class T = typename _T::T>
using Container = typename _T::template C<T>;

template <class B>
using TEST_B = BLAS1_Test<B>;

// unpacking the parameters within the test function
#define B1_TEST(name) TYPED_TEST(BLAS1_Test, name)
#define UNPACK_PARAM(test_name)    \
  using B = TypeParam;        \
  using T = typename B::type; \
  using _T = TEST_B<B>;       \
  using E = typename B::executor; \
  using test = class test_name;
#define TEST_SIZE ((option_size<test>::value == RANDOM_SIZE) ? test_size<_T>() : option_size<test>::value)
#define TEST_PREC option_prec<T, test>::value
#define EXECUTE(name)        \
  auto q = _T::make_queue(); \
  Executor<E> name(q);
#define TO_VIEW(name)                      \
  auto buf_##name = _T::make_buffer(name); \
  auto view_##name = _T::make_vview(buf_##name);

#endif
