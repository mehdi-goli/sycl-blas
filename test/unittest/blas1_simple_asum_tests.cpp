#include "blas1_simple_test.hpp"

B1_TEST(asum_simple_tests) {
  UNPACK_PARAM;
  /* ASSERT_TRUE(PERFORM(asum)({}, 0)); */
  ASSERT_TRUE(PERFORM(asum)({0}, 0));
  ASSERT_TRUE(PERFORM(asum)({1}, 1));
  ASSERT_TRUE(PERFORM(asum)({-1}, 1));
  ASSERT_TRUE(PERFORM(asum)({-1, 2, -3, 4}, 10));
  ASSERT_TRUE(PERFORM(asum)({-1, 2, -3, 4}, 10));
}
