#include "blas1_rand_test.hpp"

B1_TEST(dot_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto vX = _T::make_randcont(size),
       vY = _T::make_randcont(size),
       vR = Container<B>(1, T(0));
  // single-threaded
  T res(0);
  for(size_t i = 0; i < size; ++i)
    res += vX[i] * vY[i];
  // SYCL
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    TO_VIEW(vR);
    _dot(ex, size, view_vX, 1, view_vY, 1, view_vR);
  }
  T res2(vR[0]);
  ASSERT_TRUE(_T::eq_vals(res, res2));
}
