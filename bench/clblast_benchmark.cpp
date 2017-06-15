#include <clblast.h>

#include "benchmark.hpp"

class Context {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  bool is_active = false;

  static cl_uint get_platform_count() {
    cl_uint num_platforms;
    clGetPlatformIDs(0, NULL, &num_platforms);
    return num_platforms;
  }

  static cl_platform_id get_platform_id(size_t platform_id = 0) {
    cl_uint num_platforms = get_platform_count();
    cl_platform_id *platforms = (cl_platform_id *)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    cl_platform_id platform = platforms[platform_id];
    free(platforms);
    return platform;
  }

  static cl_uint get_device_count(cl_platform_id plat) {
    cl_uint num_devices;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    return num_devices;
  }

  static cl_device_id get_device_id(cl_platform_id plat, size_t device_id = 0) {
    cl_uint num_devices = get_device_count(plat);
    cl_device_id *devices = (cl_device_id *)malloc(num_devices * sizeof(cl_device_id));
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
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    is_active = true;
  }

  void release() {
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    is_active = false;
  }

  operator cl_context() const {
    return context;
  }

  cl_command_queue *_queue() {
    return &command_queue;
  }

  cl_command_queue queue() const {
    return command_queue;
  }

  ~Context() {
    if(is_active) {
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

  MemBuffer(ScalarT *ptr, size_t size):
    host_ptr(ptr), size(size)
  {}

  MemBuffer(size_t size, bool initialized = true):
    size(size)
  {
    private_host_ptr = true;
    host_ptr = new_data<ScalarT>(size, initialized);
  }

  ScalarT operator[](size_t i) const {
    return host_ptr[i];
  }

  ScalarT &operator[](size_t i) {
    return host_ptr[i];
  }

  cl_mem dev() {
    return dev_ptr;
  }

  ScalarT *host() {
    return host_ptr;
  }

  void create(cl_context context) {
    dev_ptr = clCreateBuffer(context, Options, size * sizeof(ScalarT), NULL, NULL);
    is_active = true;
  }

  void send(Context &ctx) {
    clEnqueueWriteBuffer(ctx.queue(), dev_ptr, CL_TRUE, 0, size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
  }

  void read(Context &ctx) {
    clEnqueueReadBuffer(ctx.queue(), dev_ptr, CL_TRUE, 0, size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
  }

  void release() {
    clReleaseMemObject(dev_ptr);
    is_active = false;
  }

  ~MemBuffer() {
    if(is_active) {
      release();
    }
    if(private_host_ptr) {
      release_data(host_ptr);
    }
  }
};

#define UNPACK_PARAM \
  using ScalarT = TypeParam; \
  Context context; \
  context.create(); \
  cl_event event = NULL;

BENCHMARK_FUNCTION(copy_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    MemBuffer<ScalarT> buf1(size), buf2(size, false);

    buf1.create(context);
    buf2.create(context);
    buf1.send(context);

    clblast::Copy<ScalarT>(size, buf1.dev(), 0, 1, buf2.dev(), 0, 1, context._queue(), &event);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Copy<ScalarT>(size, buf1.dev(), 0, 1, buf2.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), buf2.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(swap_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    MemBuffer<ScalarT> buf1(size), buf2(size);

    buf1.create(context), buf2.create(context);
    buf1.send(context), buf2.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Swap<ScalarT>(size, buf1.dev(), 0, 1, buf2.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), buf2.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(scal_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    ScalarT alpha(2.44566723436);
    MemBuffer<ScalarT> buf1(size);

    buf1.create(context);
    buf1.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Scal<ScalarT>(size, alpha, buf1.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(axpy_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    ScalarT alpha(2.44566723436);
    MemBuffer<ScalarT> buf1(size), buf2(size);

    buf1.create(context), buf2.create(context);
    buf1.send(context), buf2.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Axpy<ScalarT>(size, alpha, buf1.dev(), 0, 1, buf2.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), buf2.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(asum_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    ScalarT vr;
    MemBuffer<ScalarT> buf1(size), bufr(&vr, 1);

    buf1.create(context), bufr.create(context);
    buf1.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Asum<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), bufr.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(nrm2_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    ScalarT vr;
    MemBuffer<ScalarT> buf1(size), bufr(&vr, 1);

    buf1.create(context), bufr.create(context);
    buf1.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Nrm2<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), bufr.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(dot_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    ScalarT vr;
    MemBuffer<ScalarT> buf1(size), buf2(size), bufr(&vr, 1);

    buf1.create(context), buf2.create(context), bufr.create(context);
    buf1.send(context), buf2.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Dot<ScalarT>(size, bufr.dev(), 0, buf1.dev(), 0, 1, buf2.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), buf2.read(context), bufr.read(context);
  }
  return ms;
}

BENCHMARK_FUNCTION(iamax_bench) {
  UNPACK_PARAM;
  benchmark<>::time_units_t ms;
  {
    size_t vi;
    MemBuffer<ScalarT> buf1(size);
    MemBuffer<size_t> buf_i(&vi, 1);

    buf1.create(context), buf_i.create(context);
    buf1.send(context);

    ms = benchmark<>::duration(no_reps, [&]() {
      clblast::Iamax<ScalarT>(size, buf_i.dev(), 0, buf1.dev(), 0, 1, context._queue(), &event);
      clWaitForEvents(1, &event);
      clReleaseEvent(event);
    });

    buf1.read(context), buf_i.read(context);
  }
}

BENCHMARK_MAIN_BEGIN(1<<1, 1<<24, 10);

BENCHMARK_FLOPS(0);
BENCHMARK_REGISTER_FUNCTION("copy_float", copy_bench, float);
BENCHMARK_REGISTER_FUNCTION("copy_double", copy_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("copy_complex_float", copy_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("copy_complex_double", copy_bench, std::complex<double>); */

BENCHMARK_FLOPS(0);
BENCHMARK_REGISTER_FUNCTION("swap_float", swap_bench, float);
BENCHMARK_REGISTER_FUNCTION("swap_double", swap_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("swap_complex_float", swap_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("swap_complex_double", swap_bench, std::complex<double>); */

BENCHMARK_FLOPS(1);
BENCHMARK_REGISTER_FUNCTION("scal_float", scal_bench, float);
BENCHMARK_REGISTER_FUNCTION("scal_double", scal_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("scal_complex_float", scal_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("scal_complex_double", scal_bench, std::complex<double>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("axpy_float", axpy_bench, float);
BENCHMARK_REGISTER_FUNCTION("axpy_double", axpy_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("axpy_complex_float", axpy_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("axpy_complex_double", axpy_bench, std::complex<double>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("asum_float", asum_bench, float);
BENCHMARK_REGISTER_FUNCTION("asum_double", asum_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_float", asum_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("asum_complex_double", asum_bench, std::complex<double>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("nrm2_float", nrm2_bench, float);
BENCHMARK_REGISTER_FUNCTION("nrm2_double", nrm2_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_float", nrm2_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("nrm2_complex_double", nrm2_bench, std::complex<double>); */

BENCHMARK_FLOPS(2);
BENCHMARK_REGISTER_FUNCTION("dot_float", dot_bench, float);
BENCHMARK_REGISTER_FUNCTION("dot_double", dot_bench, double);
/* BENCHMARK_REGISTER_FUNCTION("dot_complex_float", dot_bench, std::complex<float>); */
/* BENCHMARK_REGISTER_FUNCTION("dot_complex_double", dot_bench, std::complex<double>); */

/* BENCHMARK_FLOPS(2); */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamax_float", iamax_bench, float); *1/ */
/* BENCHMARK_REGISTER_FUNCTION("iamax_double", iamax_bench, double); */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamax_complex_float", iamax_bench, std::complex<float>); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamax_complex_double", iamax_bench, std::complex<double>); *1/ */

/* BENCHMARK_FLOPS(2); */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamin_float", iamin_bench, float); *1/ */
/* BENCHMARK_REGISTER_FUNCTION("iamin_double", iamin_bench, double); */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamin_complex_float", iamin_bench, std::complex<float>); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("iamin_complex_double", iamin_bench, std::complex<double>); *1/ */

/* /1* BENCHMARK_FLOPS(0); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy2op_float", copy2op_bench, float); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy2op_double", copy2op_bench, double); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy2op_complex_float", copy2op_bench, std::complex<float>); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy2op_complex_double", copy2op_bench, std::complex<double>); *1/ */

/* /1* BENCHMARK_FLOPS(0); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy3op_float", copy3op_bench, float); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy3op_double", copy3op_bench, double); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy3op_complex_float", copy3op_bench, std::complex<float>); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("copy3op_complex_double", copy3op_bench, std::complex<double>); *1/ */

/* BENCHMARK_FLOPS(1+2+1+2+1+2+2+0); */
/* /1* BENCHMARK_REGISTER_FUNCTION("blas1_float", blas1_bench, float); *1/ */
/* BENCHMARK_REGISTER_FUNCTION("blas1_double", blas1_bench, double); */
/* /1* BENCHMARK_REGISTER_FUNCTION("blas1_complex_float", blas1_bench, std::complex<float>); *1/ */
/* /1* BENCHMARK_REGISTER_FUNCTION("blas1_complex_double", blas1_bench, std::complex<double>); *1/ */

BENCHMARK_MAIN_END();
