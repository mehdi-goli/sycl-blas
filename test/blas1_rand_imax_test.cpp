#include "blas1_rand_test.hpp"

B1_TEST(iamax_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size);
  Container<B, IndVal<T>> vI(1, IndVal<T>(
           std::numeric_limits<size_t>::max(),
           std::numeric_limits<T>::min()));
  // single-threaded
  T max = 0.;
  size_t imax = std::numeric_limits<size_t>::max();
  for(size_t i = 0; i < vX.size(); ++i)
    if(std::abs(vX[i]) > std::abs(max))
      max=vX[i], imax=i;
  IndVal<T> res(imax, max);
  // sycl
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vI);
    _iamax(ex, size, view_vX, 1, view_vI);
  }
  IndVal<T> res2(vI[0]);
  ASSERT_TRUE(_T::eq_vals(res.getVal(), res2.getVal()));
  ASSERT_EQ(res.getInd(), res2.getInd());
}
