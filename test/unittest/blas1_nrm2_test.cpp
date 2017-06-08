#include "blas1_test.hpp"

typedef ::testing::Types<blas_args<float>, blas_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, nrm2_test)
REGISTER_PREC(float, 1e-4, nrm2_test)
REGISTER_PREC(double, 1e-6, nrm2_test)
REGISTER_PREC(long double, 1e-7, nrm2_test)

B1_TEST(nrm2_simple_tests) {
  UNPACK_PARAM(nrm2_test);
  /* ASSERT_TRUE(PERFORM(nrm2)({}, 0)); */
}

B1_TEST(nrm2_test) {
  UNPACK_PARAM(nrm2_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  auto vX = _T::make_randcont(size), vR = Container<_T>(1, T(0));
  T res(0);
  for (size_t i = 0; i < size; ++i) res += vX[i] * vX[i];
  res = std::sqrt(res);

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vR);
    _nrm2(ex, size, view_vX, 1, view_vR);
  }
  ASSERT_NEAR(res, vR[0], prec);
}
