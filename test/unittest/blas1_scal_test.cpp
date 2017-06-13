#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, axpy_test)
REGISTER_PREC(float, 1e-4, axpy_test)
REGISTER_PREC(double, 1e-6, axpy_test)
REGISTER_PREC(long double, 1e-7, axpy_test)

B1_TEST(scal_test) {
  UNPACK_PARAM(axpy_test);
  size_t size = TEST_SIZE;
  ScalarT prec = TEST_PREC;

  ScalarT alpha((rand() % size * 1e2) * 1e-2);
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size, 0);
  TestClass::set_rand(vX, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    for (size_t i = 0; i < size; ++i)
      vY[i] = alpha * vX[i];

    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto view_vX = TestClass::make_vview(buf_vX);
      _scal(ex, size, alpha, view_vX, 1);
    }
    for (size_t i = 0; i < size; ++i)
      ASSERT_NEAR(vY[i], vX[i], prec);
  }
}
