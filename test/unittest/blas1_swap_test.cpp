#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, swap_test)
REGISTER_STRD(RANDOM_STRD, swap_test)

B1_TEST(swap_test) {
  UNPACK_PARAM(swap_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  std::vector<ScalarT> vZ(size);
  std::vector<ScalarT> vT(size);
  for (size_t i = 0; i < size; ++i) {
    vZ[i] = vX[i];
    vT[i] = vY[i];
  }

  bool swap_checker_mode = false;
  for (auto &d : cl::sycl::device::get_devices()) {
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vY = TestClass::make_buffer(vY);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vY = TestClass::make_vview(buf_vY);
      _swap(ex, size, view_vX, strd, view_vY, strd);
    }
    for (size_t i = 0; i < size; ++i) {
      if (i % strd == 0) {
        if (swap_checker_mode) {
          ASSERT_EQ(vZ[i], vX[i]);
          ASSERT_EQ(vT[i], vY[i]);
        } else {
          ASSERT_EQ(vZ[i], vY[i]);
          ASSERT_EQ(vT[i], vX[i]);
        }
      } else {
        ASSERT_EQ(vZ[i], vX[i]);
        ASSERT_EQ(vT[i], vY[i]);
      }
    }
    swap_checker_mode = !swap_checker_mode;
  }
}
