#include "blas1_rand_test.hpp"
#include "blas1_simple_test.hpp"

B1_TEST(asum_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size), vR = Container<B>(1, T(0));
  T res(0);
  for (auto &x : vX)
    res += std::abs(x);
  ASSERT_TRUE(PERFORM(asum)(vX, res));
}
