#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, axpy_test)
REGISTER_STRD(RANDOM_STRD, axpy_test)
REGISTER_PREC(float, 1e-4, axpy_test)
REGISTER_PREC(double, 1e-6, axpy_test)
REGISTER_PREC(long double, 1e-7, axpy_test)

B1_TEST(axpy_test) {
  UNPACK_PARAM(axpy_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  ScalarT alpha(ScalarT(rand() % size_t(size * 1e2)) * 1e-2);
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vZ(size, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    for (size_t i = 0; i < size; ++i) {
      if (i % strd == 0)
        vZ[i] = alpha * vX[i] + vY[i];
      else
        vZ[i] = vY[i];
    }

    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vY = TestClass::make_buffer(vY);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vY = TestClass::make_vview(buf_vY);
      _axpy(ex, size, alpha, view_vX, strd, view_vY, strd);
    }
    for (size_t i = 0; i < size; ++i) ASSERT_NEAR(vZ[i], vY[i], prec);
  }
}
