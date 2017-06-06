#pragma once

#include "blas1_test.hpp"

#define PERFORM(name) perform_##name<_T, T, B::template container, E>

template <class _T, class T, template <class... As> class C, class E>
bool perform_axpy(T alpha, C<T> &vX, C<T> &vY, C<T> &expect) {
  if(vX.size() != vY.size())
    throw std::domain_error("the vector sizes should be the same!");
  size_t size = vX.size();
  EXECUTE(ex) {
    TO_VIEW(vX); TO_VIEW(vY);
    _axpy<E>(ex, size, alpha, view_vX, 1, view_vY, 1);
  }
  for(size_t i = 0; i < size; ++i)
    if(!_T::eq_vals(expect[i], vY[i]))
      return false;
  return true;
}

template <class _T, class T, template <class... As> class C, class E>
bool perform_axpy(T alpha, C<T> &vX, C<T> &vY, C<T> &&expect) {
  return perform_axpy<_T, T, C, E>(alpha, vX, vY, expect);
}

template <class _T, class T, template <class... As> class C, class E>
bool perform_axpy(T alpha, C<T> &&vX, C<T> &&vY, C<T> &&expect) {
  return perform_axpy<_T, T, C, E>(alpha, vX, vY, expect);
}


template <class _T, class T, template <class... As> class C, class E>
bool perform_asum(C<T> &vX, T expect) {
  C<T> vR(1, T(0));
  size_t size = vX.size();
  EXECUTE(ex) {
    TO_VIEW(vX); TO_VIEW(vR);
    _asum(ex, size, view_vX, 1, view_vR);
  }
  return _T::eq_vals(expect, vR[0]);
}

template <class _T, class T, template <class... As> class C, class E>
bool perform_asum(C<T> &&vX, T expect) {
  return perform_asum<_T,T,C,E>(vX, expect);
}


template <class _T, class T, template <class... As> class C, class E>
bool perform_dot(C<T> &vX, C<T> &vY, T expect) {
  if(vX.size() != vY.size())
    throw std::domain_error("the vector sizes should be the same!");
  size_t size = vX.size();
  C<T> vR(1, T(0));
  EXECUTE(ex) {
    TO_VIEW(vX); TO_VIEW(vY); TO_VIEW(vR);
    _dot(ex, size, view_vX, 1, view_vY, 1, view_vR);
  }
  return _T::eq_vals(expect, vR[0]);
}

template <class _T, class T, template <class... As> class C, class E>
bool perform_dot(C<T> &&vX, C<T> &&vY, T expect) {
  return perform_dot<_T, T, C, E>(vX, vY, expect);
}


template <class _T, class T, template <class... As> class C, class E>
bool perform_nrm2(C<T> &vX, T expect) {
  size_t size = vX.size();
  C<T> vR(1, T(0));
  EXECUTE(ex) {
    TO_VIEW(vX);
    TO_VIEW(vR);
    _nrm2(ex, size, view_vX, 1, view_vR);
  }
  return _T::eq_vals(expect, vR[0]);
}

template <class _T, class T, template <class... As> class C, class E>
bool perform_nrm2(C<T> &&vX, T expect) {
  return perform_nrm2<_T, T, C, E>(vX, expect);
}
