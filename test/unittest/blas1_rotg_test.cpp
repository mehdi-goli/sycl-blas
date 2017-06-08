#include "blas1_test.hpp"

typedef ::testing::Types<
    /* blas_args<float>, */
    blas_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(2, rot_test)
REGISTER_PREC(float, 1e-4, rot_test)
REGISTER_PREC(double, 1e-7, rot_test)

B1_TEST(rot_test) {
  UNPACK_PARAM(rot_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  auto vX = _T::make_randcont(size), vY = _T::make_randcont(size);
  Container<_T> vZ(size, 0), vT(size, 0);

  T _cos, _sin;
  _T::print_cont(vX, "vX");
  _T::print_cont(vY, "vY");
  for (size_t i = 0; i < size; ++i) {
    T x = vX[i], y = vY[i];
    _rotg(x, y, _cos, _sin);
    x = vX[i], y = vY[i];
    vZ[i] = (x * _cos + y * _sin);
    vT[i] = (-x * _sin + y * _cos);
  }
  _T::print_cont(vZ, "vZ");
  _T::print_cont(vT, "vT");

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    _rot<E>(ex, size, view_vX, 1, view_vY, 1, _cos, _sin);
  }
  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(vZ[i], vX[i], prec);
    ASSERT_NEAR(vT[i], vY[i], prec);
  }
}
