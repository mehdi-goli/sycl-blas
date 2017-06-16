#include <chrono>
#include <iostream>
#include <regex>
#include <string>
#include <utility>

template <typename ScalarT>
ScalarT *new_data(size_t size, bool initialized = true) {
  ScalarT *v = new ScalarT[size];
  if (initialized) {
#pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
      v[i] = 1e-3 * ((rand() % 2000) - 1000);
    }
  }
  return v;
}
#define release_data(ptr) delete[] ptr;

struct benchmark_arguments {
  benchmark_arguments(int argc, char **argv) {}
};

template <typename time_units_t_ = std::chrono::nanoseconds,
          typename ClockT = std::chrono::system_clock>
struct benchmark {
  using time_units_t = time_units_t_;
  /**
   * @fn    duration
   * @brief Returns the duration (in chrono's type system) of the elapsed time
   */
  template <typename F, typename... Args>
  static time_units_t duration(unsigned numReps, F func, Args &&... args) {
    time_units_t dur = time_units_t::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 5; ++i) {
      func(std::forward<Args>(args)...);
    }

    unsigned reps = 0;
    for (; reps < numReps; reps++) {
      auto start = ClockT::now();

      func(std::forward<Args>(args)...);

      dur += std::chrono::duration_cast<time_units_t>(ClockT::now() - start);
    }
    return dur / reps;
  }

  static constexpr const size_t text_name_length = 30;
  static constexpr const size_t text_iterations_length = 15;
  static constexpr const size_t text_flops_length = 10;

  static std::string align_left(std::string &&text, size_t len,
                                size_t offset = 0) {
    return text + std::string((len < text.length() + offset)
                                  ? offset
                                  : len - text.length(),
                              ' ');
  }

  static void output_headers() {
    std::cout << align_left("Test", text_name_length)
              << align_left("Iterations", text_iterations_length)
              << align_left("MFlops", text_flops_length) << std::endl;
  }

  static void output_data(const std::string &short_name, int size, int no_reps,
                          int flops, time_units_t dur) {
    double sec = 1e-9 * dur.count();
    std::cout << align_left(short_name + "_" + std::to_string(size),
                            text_name_length)
              << align_left(std::to_string(no_reps), text_iterations_length)
              << align_left(std::to_string((double(size * flops) / sec) * 1e-6),
                            text_flops_length, 1)
              << "MFlops" << std::endl;
  }
};

#define BENCHMARK_FUNCTION(NAME) \
  template <class TypeParam>     \
  benchmark<>::time_units_t NAME(size_t no_reps, size_t size)

/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */
#define BENCHMARK_MAIN_BEGIN(STEP_SIZE_PARAM, NUM_STEPS, REPS) \
  int main(int argc, char *argv[]) {                           \
    benchmark_arguments ba(argc, argv);                        \
    benchmark<>::output_headers();                             \
    const unsigned num_reps = (REPS);                          \
    const unsigned step_size = (STEP_SIZE_PARAM);              \
    const unsigned max_elems = step_size * (NUM_STEPS);        \
    size_t op_flops = 0;                                       \
    {
#define BENCHMARK_FLOPS(val) op_flops = (val);
#define BENCHMARK_REGISTER_FUNCTION(NAME, FUNCTION)                          \
  for (size_t nelems = step_size; nelems < max_elems; nelems *= step_size) { \
    const std::string short_name = NAME;                                     \
    auto time = FUNCTION(num_reps, nelems);                                  \
    benchmark<>::output_data(short_name, nelems, num_reps, op_flops * 1,     \
                             time);                                          \
  }
#define BENCHMARK_MAIN_END() \
  }                          \
  }
