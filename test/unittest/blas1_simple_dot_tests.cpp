#include "blas1_simple_test.hpp"

B1_TEST(dot_simple_tests) {
  UNPACK_PARAM;
  /* ASSERT_TRUE(PERFORM(dot)({}, {}, 0)); */
  ASSERT_TRUE(PERFORM(dot)({0}, {0}, 0));
  ASSERT_TRUE(PERFORM(dot)({1}, {0}, 0));
  ASSERT_TRUE(PERFORM(dot)({0}, {1}, 0));
  ASSERT_TRUE(PERFORM(dot)({1}, {1}, 1));
  ASSERT_TRUE(PERFORM(dot)({434}, {7643}, 434*7643));
  ASSERT_TRUE(PERFORM(dot)({.434}, {764.3}, .434*764.3));
  ASSERT_TRUE(PERFORM(dot)({1,7,5,3,2}, {.3,2,.7,.4,6}, 31));
  ASSERT_TRUE(PERFORM(dot)(Container<B>(100, 1), Container<B>(100, .2), 20));
}
