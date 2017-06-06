#include "blas1_test.hpp"

#include <unistd.h>

TYPED_TEST(BLAS1_Test, simple_axpy_test) {
  UNPACK_PARAM;
  size_t size = 100;
  Container<B> vX(size, T(1)), vY(size, T(1));
  EXECUTE(executor) {
    TO_VIEW(vX); TO_VIEW(vY);
    _axpy<E>(executor, size, T(1), view_vX, 1, view_vY, 1);
  }
  auto res = std::accumulate(vY.begin(), vY.end(), T(0));
  ASSERT_TRUE(_T::eq_vals(res, T(2) * T(size)));
}
