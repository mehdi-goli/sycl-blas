#include "blas1_test.hpp"

typedef ::testing::Types<
  blas_args<float>,
  blas_args<double>
> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, iamin_test)

B1_TEST(iamin_test) {
  UNPACK_PARAM(iamin_test);
  size_t size = TEST_SIZE;

  auto vX = _T::make_randcont(size);
  Container<_T, IndVal<T>> vI(1, IndVal<T>(std::numeric_limits<size_t>::max(),
                                          std::numeric_limits<T>::max()));

  T min = std::numeric_limits<T>::max();
  size_t imin = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < vX.size(); ++i)
    if (std::abs(vX[i]) < std::abs(min)) min = vX[i], imin = i;
  IndVal<T> res(imin, min);

  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vI);
    _iamin(ex, size, view_vX, 1, view_vI);
  }
  IndVal<T> res2(vI[0]);
  ASSERT_EQ(res.getVal(), res2.getVal());
  ASSERT_EQ(res.getInd(), res2.getInd());
}
