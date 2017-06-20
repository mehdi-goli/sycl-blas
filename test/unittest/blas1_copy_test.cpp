#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, copy_test)
REGISTER_STRD(RANDOM_STRD, copy_test)

B1_TEST(copy_test) {
  UNPACK_PARAM(copy_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  for (auto &d : cl::sycl::device::get_devices()) {
    std::vector<ScalarT> vY(size, 0);
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vY = TestClass::make_buffer(vY);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vY = TestClass::make_vview(buf_vY);
      _copy(ex, size, view_vX, strd, view_vY, strd);
    }
    for (size_t i = 0; i < size; ++i) {
      if (i % strd == 0)
        ASSERT_EQ(vX[i], vY[i]);
      else
        ASSERT_EQ(0, vY[i]);
    }
  }
}
