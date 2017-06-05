#include "blas1_rand_test.hpp"

B1_TEST(iamin_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size);
  Container<B, IndVal<T>> vI(1, IndVal<T>(
           std::numeric_limits<size_t>::max(),
           std::numeric_limits<T>::max()));
  // single-threaded
  T min = 1e9;
  size_t imin = std::numeric_limits<size_t>::max();
  for(size_t i = 0; i < vX.size(); ++i)
    if(std::abs(vX[i]) < std::abs(min))
      min=vX[i], imin=i;
  IndVal<T> res(imin, min);
  // sycl
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vI);
    _iamin(ex, size, view_vX, 1, view_vI);
  }
  IndVal<T> res2(vI[0]);
  ASSERT_TRUE(_T::eq_vals(res.getVal(), res2.getVal()));
  ASSERT_EQ(res.getInd(), res2.getInd());
}
