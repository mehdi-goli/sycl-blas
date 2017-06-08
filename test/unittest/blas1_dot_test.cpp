#include "blas1_test.hpp"

typedef ::testing::Types<blas_args<float>, blas_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, dot_test)
REGISTER_PREC(float, 1e-4, dot_test)
REGISTER_PREC(double, 1e-6, dot_test)
REGISTER_PREC(long double, 1e-7, dot_test)

B1_TEST(dot_test) {
  UNPACK_PARAM(dot_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  Container<_T> vR(1, T(0));
  T res(0);
  for (size_t i = 0; i < size; ++i) res += vX[i] * vY[i];

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    TO_VIEW(vR);
    _dot(ex, size, view_vX, 1, view_vY, 1, view_vR);
  }
  ASSERT_NEAR(res, vR[0], prec);
}
