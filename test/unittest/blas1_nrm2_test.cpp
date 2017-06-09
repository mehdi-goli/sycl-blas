#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

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
  ScalarT prec = TEST_PREC;

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vR(1, 0);
  TestClass::set_rand(vX, size);

  ScalarT res(0);
  for (size_t i = 0; i < size; ++i) res += vX[i] * vX[i];
  res = std::sqrt(res);

  for (auto &d : cl::sycl::device::get_devices()) {
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vR = TestClass::make_buffer(vR);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vR = TestClass::make_vview(buf_vR);
      _nrm2(ex, size, view_vX, 1, view_vR);
    }
    ASSERT_NEAR(res, vR[0], prec);
  }
}
