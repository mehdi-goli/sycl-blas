#include "blas1_rand_test.hpp"

B1_TEST(copy_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size);
  Container<B> vY(size);
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    _copy(ex, size, view_vX, 1, view_vY, 1);
  }
  for (size_t i = 0; i < size; ++i) ASSERT_EQ(vX[i], vY[i]);
}
