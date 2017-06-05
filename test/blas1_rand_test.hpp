#pragma once

#include "blas1_test.hpp"

// to avoid annoying code duplication
#define EXECUTE(name) auto q = _T::make_queue(); Executor<E> name(q);
#define TO_VIEW(name) \
  auto buf_##name = _T::make_buffer(name); \
  auto view_##name = _T::make_vview(buf_##name);

// it is important that all tests are run with the same test size
template <class B>
size_t test_size() {
  using T = typename B::type;
  static bool first = true;
  static size_t N;
  if(first) {
    first = false;
    N = BLAS1_Test<blas_args<T>>::rand_size();
  }
  return N;
}
#define TESTSIZE(name) size_t name = test_size<B>();
