#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, asum_test)
REGISTER_STRD(RANDOM_STRD, asum_test)
REGISTER_PREC(float, 1e-4, asum_test)
REGISTER_PREC(double, 1e-6, asum_test)
REGISTER_PREC(long double, 1e-7, asum_test)

B1_TEST(asum_test) {
  UNPACK_PARAM(asum_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for(size_t i = 0; i < size; i += strd)
    res += std::abs(vX[i]);

  for (auto &d : cl::sycl::device::get_devices()) {
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vR = TestClass::make_buffer(vR);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vR = TestClass::make_vview(buf_vR);
      _asum(ex, size, view_vX, strd, view_vR);
    }
    ASSERT_NEAR(res, vR[0], prec);
  }
}
