#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, scal_test)
REGISTER_STRD(RANDOM_SIZE, scal_test)
REGISTER_PREC(float, 1e-4, scal_test)
REGISTER_PREC(double, 1e-6, scal_test)
REGISTER_PREC(long double, 1e-7, scal_test)

B1_TEST(scal_test) {
  UNPACK_PARAM(scal_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  ScalarT alpha((rand() % size * 1e2) * 1e-2);
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size, 0);
  TestClass::set_rand(vX, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    for (size_t i = 0; i < size; ++i) {
      if(i % strd == 0)
        vY[i] = alpha * vX[i];
      else
        vY[i] = vX[i];
    }

    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto view_vX = TestClass::make_vview(buf_vX);
      _scal(ex, size, alpha, view_vX, 1);
    }
    for (size_t i = 0; i < size; ++i) ASSERT_NEAR(vY[i], vX[i], prec);
  }
}
