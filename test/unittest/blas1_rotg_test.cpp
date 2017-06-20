#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(2, rot_test)
REGISTER_STRD(1, rot_test)
REGISTER_PREC(float, 1e-4, rot_test)
REGISTER_PREC(double, 1e-7, rot_test)

B1_TEST(rot_test) {
  UNPACK_PARAM(rot_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    std::vector<ScalarT> vZ(size, 0);
    std::vector<ScalarT> vT(size, 0);
    ScalarT _cos, _sin;

    for (size_t i = 0; i < size; i += strd) {
      ScalarT x = vX[i], y = vY[i];
      _rotg(x, y, _cos, _sin);
      x = vX[i], y = vY[i];
      vZ[i] = (x * _cos + y * _sin);
      vT[i] = (-x * _sin + y * _cos);
    }

    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vY = TestClass::make_buffer(vY);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vY = TestClass::make_vview(buf_vY);
      _copy(ex, size, view_vX, strd, view_vY, strd);
    }
    for (size_t i = 0; i < size; i += strd) {
      ASSERT_NEAR(vZ[i], vX[i], prec);
      ASSERT_NEAR(vT[i], vY[i], prec);
    }
  }
}
