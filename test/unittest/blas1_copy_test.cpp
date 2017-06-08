#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, copy_test)

B1_TEST(copy_test) {
  UNPACK_PARAM(copy_test);
  size_t size = TEST_SIZE;

  auto vX = TestClass::make_randcont(size);
  Container<TestClass> vY(size);
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    _copy(ex, size, view_vX, 1, view_vY, 1);
  }
  for (size_t i = 0; i < size; ++i) ASSERT_EQ(vX[i], vY[i]);
}
