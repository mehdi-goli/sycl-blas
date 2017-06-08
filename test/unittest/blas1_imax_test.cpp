#include "blas1_test.hpp"

typedef ::testing::Types<blas_args<float>, blas_args<double>> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, iamax_test)

B1_TEST(iamax_test) {
  UNPACK_PARAM(iamax_test);
  size_t size = TEST_SIZE;

  auto vX = _T::make_randcont(size);
  Container<_T, IndVal<T>> vI(1, IndVal<T>(std::numeric_limits<size_t>::max(),
                                           std::numeric_limits<T>::min()));

  T max = 0.;
  size_t imax = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < vX.size(); ++i)
    if (std::abs(vX[i]) > std::abs(max)) max = vX[i], imax = i;
  IndVal<T> res(imax, max);

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vI);
    _iamax(ex, size, view_vX, 1, view_vI);
  }
  IndVal<T> res2(vI[0]);
  ASSERT_EQ(res.getVal(), res2.getVal());
  ASSERT_EQ(res.getInd(), res2.getInd());
}
