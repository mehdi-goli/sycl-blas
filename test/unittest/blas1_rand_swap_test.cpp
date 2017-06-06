#include "blas1_rand_test.hpp"

B1_TEST(swap_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  Container<B> vZ(size, T(0)), vT(size, T(0));
  for (size_t i = 0; i < size; ++i) vZ[i] = vX[i], vT[i] = vY[i];
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    _swap(ex, size, view_vX, 1, view_vY, 1);
  }
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(vZ[i], vY[i]);
    ASSERT_EQ(vT[i], vX[i]);
  }
}
