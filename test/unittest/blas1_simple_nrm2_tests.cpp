#include "blas1_simple_test.hpp"

B1_TEST(nrm2_simple_tests) {
  UNPACK_PARAM;
  /* ASSERT_TRUE(PERFORM(nrm2)({}, 0)); */
  ASSERT_TRUE(PERFORM(nrm2)({0}, 0));
  ASSERT_TRUE(PERFORM(nrm2)({1}, 1));
  ASSERT_TRUE(PERFORM(nrm2)({1, 1}, std::sqrt(2)));
  ASSERT_TRUE(PERFORM(nrm2)({3, 4}, 5));
  ASSERT_TRUE(PERFORM(nrm2)({7, 24}, 25));
  ASSERT_TRUE(PERFORM(nrm2)(Container<B>(74233, 0), 0));
  ASSERT_TRUE(PERFORM(nrm2)(Container<B>(100, 1), 10));
}
