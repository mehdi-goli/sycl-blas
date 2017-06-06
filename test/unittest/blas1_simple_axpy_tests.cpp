#include "blas1_simple_test.hpp"

B1_TEST(axpy_simple_tests) {
  UNPACK_PARAM;
  ASSERT_TRUE(PERFORM(axpy)(287345, {}, {}, {}));
  ASSERT_TRUE(PERFORM(axpy)(2.0, {1}, {1}, {3}));
  ASSERT_TRUE(PERFORM(axpy)(M_PI, {1, 2, 3}, {1, 1, 1}, {M_PI + 1, M_PI*2 + 1, M_PI*3 + 1}));
}
