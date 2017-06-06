#include <algorithm>

#include "blas_interface_test.hpp"

B1_TEST(interface1_test) {
  UNPACK_PARAM;
  size_t size = _T::rand_size();
  std::cout << "size == "  << size << std::endl;
  auto vX = _T::make_randcont(size, {-1, 1}),
       vY = _T::make_randcont(size, {-3, 1});
  Container<B> vZ(size);
  Container<B> vR(1), vS(1), vT(1), vU(1);
  Container<B, IndVal<T>> vImax(1, IndVal<T>(std::numeric_limits<size_t>::max(),
                                             std::numeric_limits<T>::min())),
      vImin(1, IndVal<T>(std::numeric_limits<size_t>::max(),
                         std::numeric_limits<T>::max()));
  size_t imax = 0, imin = 0;
  T asum(0), alpha(0.0), dot(0), nrmX(0), nrmY(0), max(0),
      min(std::numeric_limits<T>::max()), diff(0), _cos, _sin, giv(0);
  for (size_t i = 0; i < size; ++i) {
    T &x = vX[i], &y = vY[i], &z = vZ[i];
    z = x * alpha + y, asum += std::abs(z), dot += x * z, nrmX += x * x,
    nrmY += z * z;
    if (std::abs(z) > std::abs(max)) max = z, imax = i;
    if (std::abs(z) < std::abs(min)) min = z, imin = i;
    if (i == 0) {
      T n1 = x, n2 = z;
      _rotg(n1, n2, _cos, _sin);
      diff = (z * _cos - x * _sin) - (x * _cos + z * _sin);
    } else if(i == size - 1) {
      diff += (z * _cos - x * _sin) - (x * _cos + z * _sin);
    }
    giv += ((x * _cos + z * _sin) * (z * _cos - x * _sin));
  }
  nrmX = std::sqrt(nrmX), nrmY = std::sqrt(nrmY);
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vY);
    TO_VIEW(vR);
    TO_VIEW(vS);
    TO_VIEW(vT);
    TO_VIEW(vU);
    TO_VIEW(vImax);
    TO_VIEW(vImin);

    _axpy<E>(ex, size, alpha, view_vX, 1, view_vY, 1);
    _asum<E>(ex, size, view_vY, 1, view_vR);
    _dot<E>(ex, size, view_vX, 1, view_vY, 1, view_vS);
    _nrm2<E>(ex, size, view_vY, 1, view_vT);
    _iamax<E>(ex, size, view_vY, 1, view_vImax);
    _iamin<E>(ex, size, view_vY, 1, view_vImin);
    _rot<E>(ex, size, view_vX, 1, view_vY, 1, _cos, _sin);
    _dot<E>(ex, size, view_vX, 1, view_vY, 1, view_vU);
    _swap<E>(ex, size, view_vX, 1, view_vY, 1);
  }
  T prec = std::numeric_limits<T>::epsilon() * size * 2;
  bool is_float = std::is_same<T, float>::value;
  EXPECT_TRUE(is_float && prec < 1 || prec < 1e-7);
  std::cout << "prec==" << std::fixed << std::setprecision(std::numeric_limits<T>::digits10) << prec << std::endl;
  EXPECT_TRUE(_T::eq_vals(asum, vR[0], prec));
  EXPECT_TRUE(_T::eq_vals(dot, vS[0], prec));
  EXPECT_TRUE(_T::eq_vals(nrmY, vT[0], prec));
  EXPECT_EQ(imax, vImax[0].getInd());
  EXPECT_TRUE(_T::eq_vals(max, vImax[0].getVal(), prec));
  EXPECT_EQ(imin, vImin[0].getInd());
  EXPECT_TRUE(_T::eq_vals(max, vImax[0].getVal(), prec));
  EXPECT_TRUE(_T::eq_vals(giv, vU[0], prec));
  EXPECT_TRUE(_T::eq_vals(diff, (vX[0] - vY[0]) + (vX.back() - vY.back()), prec));
}
