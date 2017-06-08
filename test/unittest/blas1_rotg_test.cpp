#include "blas1_test.hpp"

typedef ::testing::Types<
    blas1_test_args<float>,
    blas1_test_args<double>
> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(2, rot_test)
REGISTER_PREC(float, 1e-4, rot_test)
REGISTER_PREC(double, 1e-7, rot_test)

B1_TEST(rot_test) {
  UNPACK_PARAM(rot_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  auto vX = TestClass::make_randcont(size), vY = TestClass::make_randcont(size);
  Container<TestClass> vZ(size, 0), vT(size, 0);

  T _cos, _sin;
  TestClass::print_cont(vX, "vX");
  TestClass::print_cont(vY, "vY");
  for (size_t i = 0; i < size; ++i) {
    T x = vX[i], y = vY[i];
    _rotg(x, y, _cos, _sin);
    x = vX[i], y = vY[i];
    vZ[i] = (x * _cos + y * _sin);
    vT[i] = (-x * _sin + y * _cos);
  }
  TestClass::print_cont(vZ, "vZ");
  TestClass::print_cont(vT, "vT");

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
