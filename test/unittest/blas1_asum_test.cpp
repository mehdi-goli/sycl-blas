#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, asum_test)
REGISTER_PREC(float, 1e-4, asum_test)
REGISTER_PREC(double, 1e-6, asum_test)
REGISTER_PREC(long double, 1e-7, asum_test)

B1_TEST(asum_test) {
  UNPACK_PARAM(asum_test);
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  auto vX = TestClass::make_randcont(size), vR = Container<TestClass>(1, T(0));
  T res(0);
  for (auto &x : vX) res += std::abs(x);

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vR);
    _asum(ex, size, view_vX, 1, view_vR);
  }
  ASSERT_NEAR(res, vR[0], prec);
}
