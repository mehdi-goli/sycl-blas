#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double>>
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, iamin_test)
REGISTER_STRD(RANDOM_STRD, iamin_test)

B1_TEST(iamin_test) {
  UNPACK_PARAM(iamin_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  std::vector<IndVal<ScalarT>> vI(1, constant<IndVal<ScalarT>, const_val::imin>::value);

  ScalarT min = std::numeric_limits<ScalarT>::max();
  size_t imin = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) < std::abs(min)) {
      min = vX[i];
      imin = i;
    }
  }
  IndVal<ScalarT> res(imin, min);

  for (auto &d : cl::sycl::device::get_devices()) {
    auto q = TestClass::make_queue(d);
    Executor<ExecutorType> ex(q);
    {
      auto buf_vX = TestClass::make_buffer(vX);
      auto buf_vI = TestClass::make_buffer(vI);
      auto view_vX = TestClass::make_vview(buf_vX);
      auto view_vI = TestClass::make_vview(buf_vI);
      _iamin(ex, size, view_vX, strd, view_vI);
    }
    IndVal<ScalarT> res2(vI[0]);
    ASSERT_EQ(res.getVal(), res2.getVal());
    ASSERT_EQ(res.getInd(), res2.getInd());
  }
}
