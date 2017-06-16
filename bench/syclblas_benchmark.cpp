#include <interface/blas1_interface_sycl.hpp>

using namespace blas;

#include "benchmark.hpp"

template <typename DeviceSelector>
cl::sycl::queue mkqueue(DeviceSelector &s) {
  return cl::sycl::queue(s, [=](cl::sycl::exception_list eL) {
    try {
      for (auto &e : eL) std::rethrow_exception(e);
    } catch (cl::sycl::exception &e) {
      std::cout << " E " << e.what() << std::endl;
    } catch (...) {
      std::cout << " An exception " << std::endl;
    }
  });
}

template <typename ValueType>
cl::sycl::buffer<ValueType, 1> mkbuffer(ValueType *data, size_t len) {
  return cl::sycl::buffer<ValueType, 1>(data, len);
}

template <typename ValueType>
vector_view<ValueType, cl::sycl::buffer<ValueType>> mkvview(
    cl::sycl::buffer<ValueType, 1> &buf) {
  return vector_view<ValueType, cl::sycl::buffer<ValueType>>(buf);
}
#define UNPACK_PARAM              \
  using ScalarT = TypeParam;      \
  cl::sycl::default_selector dev; \
  auto q = mkqueue(dev);          \
  Executor<SYCL> ex(q);

BENCHMARK_FUNCTION(copy_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size, false);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _copy(ex, size, vvw1, 1, vvw2, 1); });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_FUNCTION(swap_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _swap(ex, size, vvw1, 1, vvw2, 1); });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_FUNCTION(scal_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT alpha(2.4367453465);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto vvw1 = mkvview(buf1);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _scal(ex, size, alpha, vvw1, 1); });
  }
  release_data(v1);
  return ms;
}

BENCHMARK_FUNCTION(axpy_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  ScalarT alpha(2.4367453465);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    ms = benchmark<>::duration(
        no_reps, [&]() { _axpy(ex, size, alpha, vvw1, 1, vvw2, 1); });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_FUNCTION(asum_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT vr;
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto bufR = mkbuffer<ScalarT>(&vr, 1);
    auto vvw1 = mkvview(buf1);
    auto vvwR = mkvview(bufR);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _asum(ex, size, vvw1, 1, vvwR); });
  }
  release_data(v1);
  return ms;
}

BENCHMARK_FUNCTION(nrm2_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT vr;
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto bufR = mkbuffer<ScalarT>(&vr, 1);
    auto vvw1 = mkvview(buf1);
    auto vvwR = mkvview(bufR);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _nrm2(ex, size, vvw1, 1, vvwR); });
  }
  release_data(v1);
  return ms;
}

BENCHMARK_FUNCTION(dot_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  ScalarT vr;
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto bufR = mkbuffer<ScalarT>(&vr, 1);
    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    auto vvwR = mkvview(bufR);
    ms = benchmark<>::duration(
        no_reps, [&]() { _dot(ex, size, vvw1, 1, vvw2, 1, vvwR); });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_FUNCTION(iamax_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  IndVal<ScalarT> vI(std::numeric_limits<size_t>::max(), 0);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf_i = mkbuffer<IndVal<ScalarT>>(&vI, 1);
    auto vvw1 = mkvview(buf1);
    auto vvw_i = mkvview(buf_i);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _iamax(ex, size, vvw1, 1, vvw_i); });
  }
  release_data(v1);
  return ms;
}

BENCHMARK_FUNCTION(iamin_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  IndVal<ScalarT> vI(std::numeric_limits<size_t>::max(), 0);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf_i = mkbuffer<IndVal<ScalarT>>(&vI, 1);
    auto vvw1 = mkvview(buf1);
    auto vvw_i = mkvview(buf_i);
    ms = benchmark<>::duration(no_reps,
                               [&]() { _iamin(ex, size, vvw1, 1, vvw_i); });
  }
  release_data(v1);
  return ms;
}

BENCHMARK_FUNCTION(scal2op_bench) {
  UNPACK_PARAM;
  ScalarT alpha(2.546562345);
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);

    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);

    ms = benchmark<>::duration(no_reps, [&]() {
      _scal(ex, size, alpha, vvw1, 1);
      _scal(ex, size, alpha, vvw2, 1);
    });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_FUNCTION(scal3op_bench) {
  UNPACK_PARAM;
  ScalarT alpha(2.546562345);
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  ScalarT *v3 = new_data<ScalarT>(size);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto buf3 = mkbuffer<ScalarT>(v3, size);

    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    auto vvw3 = mkvview(buf3);

    ms = benchmark<>::duration(no_reps, [&]() {
      _scal(ex, size, alpha, vvw1, 1);
      _scal(ex, size, alpha, vvw2, 1);
      _scal(ex, size, alpha, vvw3, 1);
    });
  }
  release_data(v1);
  release_data(v2);
  release_data(v3);
  return ms;
}

BENCHMARK_FUNCTION(blas1_bench) {
  UNPACK_PARAM;
  ScalarT *v1 = new_data<ScalarT>(size);
  ScalarT *v2 = new_data<ScalarT>(size);
  ScalarT vr[4];
  IndVal<ScalarT> vImax(std::numeric_limits<size_t>::max(), 0);
  /* IndVal<ScalarT> vImin(std::numeric_limits<size_t>::max(), 0); */
  ScalarT alpha(3.135345123);
  benchmark<>::time_units_t ms;
  {
    auto buf1 = mkbuffer<ScalarT>(v1, size);
    auto buf2 = mkbuffer<ScalarT>(v2, size);
    auto bufr0 = mkbuffer<ScalarT>(&vr[0], 1);
    auto bufr1 = mkbuffer<ScalarT>(&vr[1], 1);
    auto bufr2 = mkbuffer<ScalarT>(&vr[2], 1);
    auto bufr3 = mkbuffer<ScalarT>(&vr[3], 1);
    auto buf_i1 = mkbuffer<IndVal<ScalarT>>(&vImax, 1);
    /* auto buf_i2 = mkbuffer<IndVal<ScalarT>>(&vImin, 1); */

    auto vvw1 = mkvview(buf1);
    auto vvw2 = mkvview(buf2);
    auto vvwr0 = mkvview(bufr0);
    auto vvwr1 = mkvview(bufr1);
    auto vvwr2 = mkvview(bufr2);
    auto vvwr3 = mkvview(bufr3);
    auto vvw_i1 = mkvview(buf_i1);
    /* auto vvw_i2 = mkvview(buf_i2); */

    ms = benchmark<>::duration(no_reps, [&]() {
      _axpy(ex, size, alpha, vvw1, 1, vvw2, 1);
      _asum(ex, size, vvw2, 1, vvwr0);
      _dot(ex, size, vvw1, 1, vvw2, 1, vvwr1);
      _nrm2(ex, size, vvw2, 1, vvwr2);
      _iamax(ex, size, vvw2, 1, vvw_i1);
      /* _iamin(ex, size, vvw2, 1, vvw_i2); */
      _dot(ex, size, vvw1, 1, vvw2, 1, vvwr3);
      _swap(ex, size, vvw1, 1, vvw2, 1);
    });
  }
  release_data(v1);
  release_data(v2);
  return ms;
}

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 24, 10);

/* BENCHMARK_FLOPS(0); */
/* BENCHMARK_REGISTER_FUNCTION("copy_float", copy_bench<float>); */
/* BENCHMARK_REGISTER_FUNCTION("copy_double", copy_bench<double>); */
/* BENCHMARK_REGISTER_FUNCTION("copy_complex_float",
 * copy_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("copy_complex_double",
 * copy_bench<std::complex<double>>); */

/* BENCHMARK_FLOPS(0); */
/* BENCHMARK_REGISTER_FUNCTION("swap_float", swap_bench<float>); */
/* BENCHMARK_REGISTER_FUNCTION("swap_double", swap_bench<double>); */
/* BENCHMARK_REGISTER_FUNCTION("swap_complex_float",
 * swap_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("swap_complex_double",
 * swap_bench<std::complex<double>>); */

BENCHMARK_FLOPS(1);
BENCHMARK_REGISTER_FUNCTION("scal_float", scal_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal_double", scal_bench<double>);
BENCHMARK_REGISTER_FUNCTION("scal_complex_float",
                            scal_bench<std::complex<float>>);
BENCHMARK_REGISTER_FUNCTION("scal_complex_double",
                            scal_bench<std::complex<double>>);

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("axpy_float", axpy_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy_double", axpy_bench<double>);
BENCHMARK_REGISTER_FUNCTION("axpy_complex_float",
                            axpy_bench<std::complex<float>>);
BENCHMARK_REGISTER_FUNCTION("axpy_complex_double",
                            axpy_bench<std::complex<double>>);

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("asum_float", asum_bench<float>);
BENCHMARK_REGISTER_FUNCTION("asum_double", asum_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_float",
 * asum_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_double",
 * asum_bench<std::complex<double>>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("nrm2_float", nrm2_bench<float>);
BENCHMARK_REGISTER_FUNCTION("nrm2_double", nrm2_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_float",
 * nrm2_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_double",
 * nrm2_bench<std::complex<double>>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("dot_float", dot_bench<float>);
BENCHMARK_REGISTER_FUNCTION("dot_double", dot_bench<double>);
BENCHMARK_REGISTER_FUNCTION("dot_complex_float",
                            dot_bench<std::complex<float>>);
BENCHMARK_REGISTER_FUNCTION("dot_complex_double",
                            dot_bench<std::complex<double>>);

BENCHMARK_FLOPS(2);
/* BENCHMARK_REGISTER_FUNCTION("iamax_float", iamax_bench<float>); */
BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("iamax_complex_float",
 * iamax_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("iamax_complex_double",
 * iamax_bench<std::complex<double>>); */

/* BENCHMARK_FLOPS(2); */
/* BENCHMARK_REGISTER_FUNCTION("iamin_float", iamin_bench<float>); */
BENCHMARK_REGISTER_FUNCTION("iamin_double", iamin_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("iamin_complex_float",
 * iamin_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("iamin_complex_double",
 * iamin_bench<std::complex<double>>); */

BENCHMARK_FLOPS(1);
BENCHMARK_REGISTER_FUNCTION("scal2op_float", scal2op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal2op_double", scal2op_bench<double>);
BENCHMARK_REGISTER_FUNCTION("scal2op_complex_float",
                            scal2op_bench<std::complex<float>>);
BENCHMARK_REGISTER_FUNCTION("scal2op_complex_double",
                            scal2op_bench<std::complex<double>>);

BENCHMARK_FLOPS(1);
BENCHMARK_REGISTER_FUNCTION("scal3op_float", scal3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal3op_double", scal3op_bench<double>);
BENCHMARK_REGISTER_FUNCTION("scal3op_complex_float",
                            scal3op_bench<std::complex<float>>);
BENCHMARK_REGISTER_FUNCTION("scal3op_complex_double",
                            scal3op_bench<std::complex<double>>);

BENCHMARK_FLOPS(2 + 2 + 2 + 2 + 2 + 2 + 2 + 0);
/* BENCHMARK_REGISTER_FUNCTION("blas1_float", blas1_bench<float>); */
BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("blas1_complex_float",
 * blas1_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("blas1_complex_double",
 * blas1_bench<std::complex<double>>); */

BENCHMARK_MAIN_END();
