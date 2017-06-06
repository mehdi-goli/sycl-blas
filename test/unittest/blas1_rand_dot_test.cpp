#include "blas1_rand_test.hpp"
#include "blas1_simple_test.hpp"

B1_TEST(dot_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  T res(0);
  for (size_t i = 0; i < size; ++i)
    res += vX[i] * vY[i];
  ASSERT_TRUE(PERFORM(dot)(vX, vY, res));
}
