#include "blas1_test.hpp"

typedef ::testing::Types<blas_args<float>, blas_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, axpy_test)
REGISTER_PREC(float, 1e-4, axpy_test)
REGISTER_PREC(double, 1e-6, axpy_test)
REGISTER_PREC(long double, 1e-7, axpy_test)

B1_TEST(axpy_test) {
  UNPACK_PARAM(axpy_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  T alpha((rand() % size * 1e2) * 1e-2);
  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  Container<_T> vZ(size, 0);
  for (size_t i = 0; i < size; ++i) vZ[i] = alpha * vX[i] + vY[i];

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    _axpy<E>(ex, size, alpha, view_vX, 1, view_vY, 1);
  }
  for (size_t i = 0; i < size; ++i) ASSERT_NEAR(vY[i], vZ[i], prec);
}
