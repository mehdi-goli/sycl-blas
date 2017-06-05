#include "blas1_test.hpp"

#include <unistd.h>

TYPED_TEST(BLAS1_Test, simple_axpy_test) {
  size_t size = 100;
  using B = TypeParam;
  using T = typename B::type;
  using _T = TEST_B<B>;
  Container<B> vX(size, T(1)), vY(size, T(1));
  auto q = _T::make_queue();
  Executor<SYCL> executor(q);
  {
    auto bX = _T::make_buffer(vX), bY = _T::make_buffer(vY);
    auto view_x = _T::make_vview(bX), view_y = _T::make_vview(bY);
    _axpy<SYCL>(executor, bX.get_count(), 1.0, view_x, 1, view_y, 1);
  }
  auto res = std::accumulate(vY.begin(), vY.end(), T(0));
  ASSERT_TRUE(_T::eq_vals(res, T(2) * T(size)));
}
