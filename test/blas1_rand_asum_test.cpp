#include "blas1_rand_test.hpp"

B1_TEST(asum_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  auto
    vX = _T::make_randcont(size),
    vR = Container<B>(1, T(0));
  // single-threaded
  T res(0);
  for(auto &x:vX)
    res += std::abs(x);
  // sycl
  EXECUTE(ex) {
    TO_VIEW(vX); TO_VIEW(vR);
    _asum<E>(ex, size, view_vX, 1, view_vR);
  }
  T res2(vR[0]);
  ASSERT_TRUE(_T::eq_vals(res, res2));
}
