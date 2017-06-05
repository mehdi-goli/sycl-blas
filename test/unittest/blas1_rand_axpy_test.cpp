#include "blas1_rand_test.hpp"

B1_TEST(axpy_test) {
  UNPACK_PARAM;
  TESTSIZE(size);
  T alpha(rand() % size);
  auto vX = _T::make_randcont(size),
       vY = _T::make_randcont(size);
  // single-threaded:
  Container<B> vZ(size, 0);
  for(size_t i = 0; i < size; ++i)
    vZ[i] = alpha * vX[i] + vY[i];
  // SYCL
  EXECUTE(ex) {
    TO_VIEW(vX); TO_VIEW(vY);
    _axpy<E>(ex, size, alpha, view_vX, 1, view_vY, 1);
  }
  for(size_t i = 0; i < size; ++i)
    ASSERT_TRUE(_T::eq_vals(vY[i], vZ[i]));
}
