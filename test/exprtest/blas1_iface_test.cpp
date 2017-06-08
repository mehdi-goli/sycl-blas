#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<double>> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(RANDOM_SIZE, interface1_test)
REGISTER_PREC(float, 1e-4, interface1_test)
REGISTER_PREC(double, 1e-6, interface1_test)

B1_TEST(interface1_test) {
  UNPACK_PARAM(interface1_test)
  size_t size = TEST_SIZE;
  T prec = TEST_PREC;

  std::cout << "size == " << size << std::endl;
  auto vX = TestClass::make_randcont(size, {-1, 1}),
       vY = TestClass::make_randcont(size, {-3, 1});
  Container<TestClass> vZ(size);
  Container<TestClass> vR(1), vS(1), vT(1), vU(1);
  Container<TestClass, IndVal<T>> vImax(1,
                                 IndVal<T>(std::numeric_limits<size_t>::max(),
                                           std::numeric_limits<T>::min())),
      vImin(1, IndVal<T>(std::numeric_limits<size_t>::max(),
                         std::numeric_limits<T>::max()));
  size_t imax = 0, imin = 0;
  T asum(0), alpha(0.0), dot(0), nrmX(0), nrmY(0), max(0),
      min(std::numeric_limits<T>::max()), diff(0), _cos(0), _sin(0), giv(0);
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
    } else if (i == size - 1) {
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
  T prec_sample =
      std::max(std::numeric_limits<T>::epsilon() * size * 2, prec * T(1e1));
  EXPECT_LE(prec_sample, prec * 1e4);
  std::cout << "prec==" << std::fixed
            << std::setprecision(std::numeric_limits<T>::digits10)
            << prec_sample << std::endl;
  EXPECT_NEAR(asum, vR[0], prec_sample);
  EXPECT_NEAR(dot, vS[0], prec_sample);
  EXPECT_NEAR(nrmY, vT[0], prec_sample);
  EXPECT_EQ(imax, vImax[0].getInd());
  EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
  EXPECT_EQ(imin, vImin[0].getInd());
  EXPECT_NEAR(max, vImax[0].getVal(), prec_sample);
  /* EXPECT_NEAR(giv, vU[0], prec_sample); */
  EXPECT_NEAR(diff, (vX[0] - vY[0]) + (vX.back() - vY.back()), prec_sample);
}
