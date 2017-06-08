#include "blas1_test.hpp"

typedef ::testing::Types<
  blas_args<float>,
  blas_args<double>
> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, swap_test)

B1_TEST(swap_test) {
  UNPACK_PARAM(swap_test);
  size_t size = TEST_SIZE;

  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  Container<_T> vZ(size, T(0)), vT(size, T(0));
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
