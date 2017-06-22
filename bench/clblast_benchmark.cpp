#include <complex>
#include <vector>

#include <clblast.h>

#include "benchmark.hpp"

class Context {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  bool is_active = false;

  static cl_uint get_platform_count() {
    cl_uint num_platforflops;
    clGetPlatformIDs(0, NULL, &num_platforflops);
    return num_platforflops;
  }

  static cl_platform_id get_platform_id(size_t platform_id = 0) {
    cl_uint num_platforflops = get_platform_count();
    cl_platform_id *platforflops =
        (cl_platform_id *)malloc(num_platforflops * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforflops, platforflops, NULL);
    cl_platform_id platform = platforflops[platform_id];
    free(platforflops);
    return platform;
  }

  static cl_uint get_device_count(cl_platform_id plat) {
    cl_uint num_devices;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    return num_devices;
  }

  static cl_device_id get_device_id(cl_platform_id plat, size_t device_id = 0) {
    cl_uint num_devices = get_device_count(plat);
    cl_device_id *devices =
        (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    cl_device_id device = devices[device_id];
    free(devices);
    return device;
  }

 public:
  Context(size_t plat_id = 0, size_t dev_id = 0) {
    platform = get_platform_id(plat_id);
    device = get_device_id(platform, dev_id);
  }

  void create() {
    if (is_active) {
      throw std::runtime_error("context is already active");
    }
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    is_active = true;
  }

  void release() {
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    is_active = false;
  }

  operator cl_context() const { return context; }

  cl_command_queue *_queue() { return &command_queue; }

  cl_command_queue queue() const { return command_queue; }

  ~Context() {
    if (is_active) {
      release();
    }
  }
};

template <typename ScalarT, int Options = CL_MEM_READ_WRITE>
class MemBuffer {
  size_t size = 0;
  cl_mem dev_ptr = NULL;
  ScalarT *host_ptr = NULL;
  bool private_host_ptr = false;
  bool is_active = false;

 public:
  MemBuffer(ScalarT *ptr, size_t size) : host_ptr(ptr), size(size) {}

  MemBuffer(size_t size, bool initialized = true) : size(size) {
    private_host_ptr = true;
    host_ptr = new_data<ScalarT>(size, initialized);
  }

  ScalarT operator[](size_t i) const { return host_ptr[i]; }

  ScalarT &operator[](size_t i) { return host_ptr[i]; }

  cl_mem dev() { return dev_ptr; }

  ScalarT *host() { return host_ptr; }

  void create(cl_context context) {
    if (is_active) {
      throw std::runtime_error("buffer is already active");
    }
    dev_ptr =
        clCreateBuffer(context, Options, size * sizeof(ScalarT), NULL, NULL);
    is_active = true;
  }

  void send(Context &ctx) {
    if (!is_active) {
      create(ctx);
    }
    clEnqueueWriteBuffer(ctx.queue(), dev_ptr, CL_TRUE, 0,
                         size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
  }

  void read(Context &ctx) {
    clEnqueueReadBuffer(ctx.queue(), dev_ptr, CL_TRUE, 0,
                        size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
  }

  void release() {
    if (!is_active) {
      throw std::runtime_error("cannot release inactive buffer");
    }
    clReleaseMemObject(dev_ptr);
    is_active = false;
  }

  ~MemBuffer() {
    if (is_active) {
      release();
    }
    if (private_host_ptr) {
      release_data(host_ptr);
    }
  }
};

#define UNPACK_PARAM         \
  using ScalarT = TypeParam; \
  cl_event event = NULL;

class ClBlastBenchmarker {
  Context context;

 public:
  ClBlastBenchmarker() : context() { context.create(); }

  BENCHMARK_FUNCTION(scal_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(size);

      buf1.send(context);

      flops = benchmark<>::measure(no_reps, size * 1, [&]() {
        clblast::Scal<ScalarT>(size, alpha, buf1.dev(), 0, 1, context._queue(),
                               &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(axpy_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> buf2(size);

      buf1.send(context);
      buf2.send(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Axpy<ScalarT>(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0, 1,
                               context._queue(), &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context), buf2.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(asum_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> bufr(&vr, 1);

      buf1.send(context);
      bufr.create(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Asum<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                               context._queue(), &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context), bufr.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(nrm2_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> bufr(&vr, 1);

      buf1.send(context);
      bufr.create(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Nrm2<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1,
                               context._queue(), &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context), bufr.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(dot_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT vr;
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> buf2(size);
      MemBuffer<ScalarT> bufr(&vr, 1);

      buf1.send(context);
      buf2.send(context);
      bufr.create(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Dot<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1, buf2.dev(),
                              0, 1, context._queue(), &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context);
      buf2.read(context);
      bufr.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(iamax_bench) {
    UNPACK_PARAM;
    double flops;
    {
      int vi;
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<int> buf_i(&vi, 1);

      buf1.send(context);
      buf_i.create(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Amax<ScalarT>(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                               context._queue(), &event);
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
      });

      buf1.read(context);
      buf_i.read(context);
    }
    return flops;
  }

  // not supported at current release yet
  /* BENCHMARK_FUNCTION(iamin_bench) { */
  /*   UNPACK_PARAM; */
  /*   double flops; */
  /*   { */
  /*     int vi; */
  /*     MemBuffer<ScalarT> buf1(size); */
  /*     MemBuffer<int> buf_i(&vi, 1); */

  /*     buf1.create(context); */
  /*     buf_i.create(context); */
  /*     buf1.send(context); */

  /*     flops = benchmark<>::measure(no_reps, [&]() { */
  /*       clblast::Amin<ScalarT>(size, buf_i.dev(), 0, buf1.dev(), 0, 1, */
  /*                               context._queue(), &event); */
  /*       clWaitForEvents(1, &event); */
  /*       clReleaseEvent(event); */
  /*     }); */

  /*     buf1.read(context), buf_i.read(context); */
  /*   } */
  /*   return flops; */
  /* } */

  BENCHMARK_FUNCTION(scal2op_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> buf2(size);

      buf1.send(context);
      buf2.send(context);

      flops = benchmark<>::measure(no_reps, size * 2, [&]() {
        clblast::Scal<ScalarT>(size, alpha, buf1.dev(), 0, 1, context._queue(),
                               &event);
        clblast::Scal<ScalarT>(size, alpha, buf2.dev(), 0, 1, context._queue(),
                               &event);
      });

      buf1.read(context);
      buf2.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(scal3op_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alpha(2.4367453465);
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> buf2(size);
      MemBuffer<ScalarT> buf3(size);

      buf1.send(context);
      buf2.send(context);
      buf3.send(context);

      flops = benchmark<>::measure(no_reps, size * 3, [&]() {
        clblast::Scal<ScalarT>(size, alpha, buf1.dev(), 0, 1, context._queue(), &event);
        clblast::Scal<ScalarT>(size, alpha, buf2.dev(), 0, 1, context._queue(), &event);
        clblast::Scal<ScalarT>(size, alpha, buf3.dev(), 0, 1, context._queue(), &event);
      });

      buf1.read(context);
      buf2.read(context);
      buf3.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(axpy3op_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alphas[] = { 1.78426458744, 2.187346575843, 3.78164387328 };
      size_t offsets[] = { 0, size, size * 2 };
      MemBuffer<ScalarT> bufsrc(size * 3);
      MemBuffer<ScalarT> bufdst(size * 3);

      bufsrc.send(context);
      bufdst.send(context);

      flops = benchmark<>::measure(no_reps, size * 3 * 2, [&](){
        clblast::AxpyBatched<ScalarT>(size, alphas, bufsrc.dev(), offsets, 1, bufdst.dev(), offsets, 1, 3, context._queue(), &event);
      });

      bufsrc.read(context);
      bufdst.read(context);
    }
    return flops;
  }

  BENCHMARK_FUNCTION(blas1_bench) {
    UNPACK_PARAM;
    double flops;
    {
      ScalarT alpha(3.135345123);
      MemBuffer<ScalarT> buf1(size);
      MemBuffer<ScalarT> buf2(size);
      ScalarT vr[4];
      size_t vi;
      MemBuffer<ScalarT> bufr(vr, 4);
      MemBuffer<size_t> buf_i(&vi, 1);

      buf1.send(context);
      buf2.send(context);
      bufr.create(context);
      buf_i.create(context);

      flops = benchmark<>::measure(no_reps, size * 12, [&]() {
        clblast::Axpy<ScalarT>(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0, 1,
                               context._queue(), &event);
        clblast::Asum<ScalarT>(size, bufr.dev(), 0, buf2.dev(), 0, 1,
                               context._queue(), &event);
        clblast::Dot<ScalarT>(size, bufr.dev(), 1, buf1.dev(), 0, 1, buf2.dev(),
                              0, 1, context._queue(), &event);
        clblast::Nrm2<ScalarT>(size, bufr.dev(), 2, buf1.dev(), 0, 1,
                               context._queue(), &event);
        clblast::Amax<ScalarT>(size, buf_i.dev(), 0, buf1.dev(), 0, 1,
                               context._queue(), &event);
        clblast::Swap<ScalarT>(size, buf1.dev(), 0, 1, buf2.dev(), 0, 1,
                               context._queue(), &event);
      });
      buf1.read(context);
      buf2.read(context);
      bufr.read(context);
      buf_i.read(context);
    }
    return flops;
  }
};

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 24, 10);
ClBlastBenchmarker blasbenchmark;

BENCHMARK_REGISTER_FUNCTION("scal_float", scal_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal_double", scal_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("scal_complex_float", */
/*                             scal_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("scal_complex_double", */
/*                             scal_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("axpy_float", axpy_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy_double", axpy_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("axpy_complex_float", */
/*                             axpy_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("axpy_complex_double", */
/*                             axpy_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("asum_float", asum_bench<float>);
BENCHMARK_REGISTER_FUNCTION("asum_double", asum_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_float", */
/*                             asum_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_double", */
/*                             asum_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("nrm2_float", nrm2_bench<float>);
BENCHMARK_REGISTER_FUNCTION("nrm2_double", nrm2_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_float", */
/*                             nrm2_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_double", */
/*                             nrm2_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("dot_float", dot_bench<float>);
BENCHMARK_REGISTER_FUNCTION("dot_double", dot_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("dot_complex_float", */
/*                             dot_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("dot_complex_double", */
/*                             dot_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("iamax_float", iamax_bench<float>);
BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("iamax_complex_float", */
/*                             iamax_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("iamax_complex_double", */
/*                             iamax_bench<std::complex<double>>); */

/* BENCHMARK_REGISTER_FUNCTION("iamin_float", iamin_bench<float>); */
/* BENCHMARK_REGISTER_FUNCTION("iamin_double", iamin_bench<double>); */
/* BENCHMARK_REGISTER_FUNCTION("iamin_complex_float", */
/*                             iamin_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("iamin_complex_double", */
/*                             iamin_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("scal2op_float", scal2op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal2op_double", scal2op_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("scal2op_complex_float", */
/*                             scal2op_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("scal2op_complex_double", */
/*                             scal2op_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("scal3op_float", scal3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("scal3op_double", scal3op_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("scal3op_complex_float", */
/*                             scal3op_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("scal3op_complex_double", */
/*                             scal3op_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("axpy3op_float", axpy3op_bench<float>);
BENCHMARK_REGISTER_FUNCTION("axpy3op_double", axpy3op_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("axpy3op_complex_float", */
/*                             axpy3op_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("axpy3op_complex_double", */
/*                             axpy3op_bench<std::complex<double>>); */

BENCHMARK_REGISTER_FUNCTION("blas1_float", blas1_bench<float>);
BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench<double>);
/* BENCHMARK_REGISTER_FUNCTION("blas1_complex_float", */
/*                             blas1_bench<std::complex<float>>); */
/* BENCHMARK_REGISTER_FUNCTION("blas1_complex_double", */
/*                             blas1_bench<std::complex<double>>); */

BENCHMARK_MAIN_END();
