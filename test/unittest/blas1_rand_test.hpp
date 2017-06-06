#pragma once

#include "blas1_test.hpp"

// it is important that all tests are run with the same test size
template <class B>
size_t test_size() {
  /* return 1e6; */
  using T = typename B::type;
  static bool first = true;
  static size_t N;
  if (first) {
    first = false;
    N = BLAS1_Test<blas_args<T>>::rand_size();
  }
  return N;
}
#define TESTSIZE(name) size_t name = test_size<B>();
