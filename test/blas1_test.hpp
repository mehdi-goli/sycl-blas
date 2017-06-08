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

template <typename C>
struct option_size;
#define RANDOM_SIZE UINT_MAX
#define REGISTER_SIZE(size, test_class)         \
  template <>                                   \
  struct option_size<class test_class> {        \
    static constexpr const size_t value = size; \
  };
template <class T, typename C>
struct option_prec;
#define REGISTER_PREC(type, val, Test_Class)   \
  template <>                                  \
  struct option_prec<type, class Test_Class> { \
    static constexpr const type value = val;   \
  };

/*
 * Template arguments:
 * T = (scalar) type
 * C = container
 * E = executor
 */

// Wraps the above arguments into one template parameter.
// We will treat template-specialized blas_templ_struct as a single class
template <class T, template <class... As> class C, class E>
struct blas_templ_struct {
  using type = T;
  template <class U = T>
  using container = C<U>;
  using executor = E;
};
// A "using" shortcut for the struct
template <class T, template <class... As> class C = std::vector, class E = SYCL>
using blas1_test_args = blas_templ_struct<T, C, E>;

// the test class itself
template <class B> class BLAS1_Test;

template <class T_, template <class... As> class C_, class E_>
class BLAS1_Test<blas1_test_args<T_, C_, E_>> : public ::testing::Test {
 public:
  using T = T_;
  template <class U = T>
  using C = C_<U>;
  using E = E_;

  BLAS1_Test() {}

  virtual ~BLAS1_Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  template <class U = T>
  static size_t rand_size() {
    size_t ret = rand() >> 5;
    int type_size = sizeof(U) * CHAR_BIT - std::numeric_limits<U>::digits10 - 2;
    return (ret & (std::numeric_limits<size_t>::max() +
                   (size_t(1) << (type_size - 2)))) +
           1;
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
// so each time we access this function within the same program, we get the same
// randomly generated size
template <class TestClass>
size_t test_size() {
  using T = typename TestClass::T;
  static bool first = true;
  static size_t N;
  if (first) {
    first = false;
    N = TestClass::rand_size();
  }
  return N;
}

// templates the container type of the test class
// TestClass - e.g. BLAS1_Test
// T - element (or scalar) type, by default is TestClass::T
template <class TestClass, class T = typename TestClass::T>
using Container = typename TestClass::template C<T>;

// unpacking the parameters within the test function
// B is blas_templ_struct
// TestClass is BLAS1_Test<B>
// T is default (scalar) type for the test (e.g. float, double)
// C is the container type for the test (e.g. std::vector)
// E is the executor kind for the test (sequential, openmp, sycl)
#define B1_TEST(name) TYPED_TEST(BLAS1_Test, name)
#define UNPACK_PARAM(test_name)   \
  using B = TypeParam;            \
  using T = typename B::type;     \
  using TestClass = BLAS1_Test<B>;           \
  using E = typename B::executor; \
  using test = class test_name;
// TEST_SIZE determines the size based on the suggestion
#define TEST_SIZE                                              \
  ((option_size<test>::value == RANDOM_SIZE) ? test_size<TestClass>() \
                                             : option_size<test>::value)
// TEST_PREC determines the precision for the test based on the suggestion for
// the type
#define TEST_PREC option_prec<T, test>::value
// a shortcut for creating a queue and an executor of that queue
#define EXECUTE(name)        \
  auto q = TestClass::make_queue(); \
  Executor<E> name(q);
// a shortcut for creating a buffer from a vector and a vector_view from that
// buffer
#define TO_VIEW(name)                      \
  auto buf_##name = TestClass::make_buffer(name); \
  auto view_##name = TestClass::make_vview(buf_##name);

#endif
