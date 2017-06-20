#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double>>
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, iamax_test)
REGISTER_STRD(RANDOM_STRD, iamax_test)

B1_TEST(iamax_test) {
  UNPACK_PARAM(iamax_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  std::vector<IndVal<ScalarT>> vI(1, constant<IndVal<ScalarT>, const_val::imax>::value);

  ScalarT max = 0.;
  size_t imax = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndVal<ScalarT> res(imax, max);

  for (auto &d : cl::sycl::device::get_devices()) {
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vI = TestClass::make_buffer(vI);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vI = TestClass::make_vview(buf_vI);
      _iamax(ex, size, view_vX, strd, view_vI);
    }
    IndVal<ScalarT> res2(vI[0]);
    ASSERT_EQ(res.getVal(), res2.getVal());
    ASSERT_EQ(res.getInd(), res2.getInd());
  }
}
