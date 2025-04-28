// Main benchmark function of MMul

#include "benchmark/benchmark.h"

#include "../platform.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#ifdef _WIN32
#define aligned_alloc _aligned_malloc
#endif

unsigned int numThreads = std::thread::hardware_concurrency();

// Blocked column multi-output serial implementation
void blocked_column_multi_output_mmul(const float* A, const float* B, float* C, std::size_t N)
{
  // For each chunk of columns
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += 16)
  // For each chunk of rows
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
    // For each block of elements in this row of this column chunk
    // Solve for 16 elements at a time
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
      for (std::size_t tile = 0; tile < N; tile += 16)
      // Apply that tile to each row of the row chunk
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
        for (std::size_t row = 0; row < 16; row++)
        // For each row in the tile
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
          // Solve for each element in this tile row
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
            for (std::size_t idx = 0; idx < 16; idx++)
              C[(row + row_chunk) * N + col_chunk + idx] +=
                A[(row + row_chunk) * N + tile + tile_row] *
                B[tile * N + tile_row * N + col_chunk + idx];
}
// Blocked column multi-output MMul with aligned memory benchmark
static void blocked_column_multi_output_aligned_mmul_bench(benchmark::State& s)
{
  // Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Create input matrices
  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  // Main benchmark loop
  for (auto _ : s)
  {
    blocked_column_multi_output_mmul(A, B, C, N);
  }

  // Free memory
  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}
BENCHMARK(blocked_column_multi_output_aligned_mmul_bench)
  ->Arg(1 * BENCH_SCALE * numThreads * 16)
  ->Arg(2 * BENCH_SCALE * numThreads * 16)
  ->Arg(3 * BENCH_SCALE * numThreads * 16)
  ->Unit(benchmark::kMillisecond);

// Blocked column parallel implementation w/o atomic
void blocked_column_multi_output_parallel_mmul(const float* A, const float* B, float* C,
  std::size_t N, std::size_t start_col, std::size_t end_col)
{
  // For each chunk of columns
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
  for (std::size_t col_chunk = start_col; col_chunk < end_col; col_chunk += 16)
  // For each chunk of rows
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
    // For each block of elements in this row of this column chunk
    // Solve for 16 elements at a time
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
      for (std::size_t tile = 0; tile < N; tile += 16)
      // For apply that tile to each row of the row chunk
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
        for (std::size_t row = 0; row < 16; row++)
        // For each row in the tile
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
          // Solve for each element in this tile row
#pragma loop(ivdep)
#ifdef _MSC_VER
#pragma vector aligned
#endif
            for (std::size_t idx = 0; idx < 16; idx++)
              C[(row + row_chunk) * N + col_chunk + idx] +=
                A[(row + row_chunk) * N + tile + tile_row] *
                B[tile * N + tile_row * N + col_chunk + idx];
}

#if 0
static void parallel_blocked_column_multi_output_mmul_bench(benchmark::State& s)
{
  // Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Create input matrices
  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  // Initialize them
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  std::size_t num_threads = numThreads;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Allocate per-thread private C buffers
  std::vector<float*> thread_C_buffers(num_threads);
  for (std::size_t i = 0; i < num_threads; i++)
    thread_C_buffers[i] = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  std::size_t n_cols = N / num_threads;

  for (auto _ : s)
  {
    // Clear per-thread buffers
    for (std::size_t i = 0; i < num_threads; i++)
      std::fill(thread_C_buffers[i], thread_C_buffers[i] + N * N, 0.0f);

    // Launch threads
    std::size_t start_col = 0;
    for (std::size_t i = 0; i < num_threads; i++)
    {
      auto end_col = start_col + n_cols;
      float* local_C = thread_C_buffers[i];
      threads.emplace_back(
        [=] { blocked_column_multi_output_parallel_mmul(A, B, local_C, N, start_col, end_col); });
      start_col += n_cols;
    }

    // Wait for all threads
    for (auto& t : threads)
      t.join();
    threads.clear();

    // Reduction: sum all thread local_C buffers into C
    for (std::size_t i = 0; i < num_threads; i++)
    {
      float* local_C = thread_C_buffers[i];
      for (std::size_t idx = 0; idx < N * N; idx++)
        C[idx] += local_C[idx];
    }
  }

  // Free memory
  for (auto buf : thread_C_buffers)
    ALIGNED_FREE(buf);

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}
#else
// Parallel blocked column multi-output MMul benchmark
static void parallel_blocked_column_multi_output_mmul_bench(benchmark::State& s)
{
  // Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Create input matrices
  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  // Set up for launching threads
  std::size_t num_threads = numThreads;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Calculate values to pass to threads
  // Assumed to be divisable by num_threads (evenly)
  std::size_t n_cols = N / num_threads;

  // Main benchmark loop
  for (auto _ : s)
  {
    // Launch threads
    std::size_t start_col = 0;
    for (std::size_t i = 0; i < num_threads; i++)
    {
      auto end_col = start_col + n_cols;
      threads.emplace_back(
        [&] { blocked_column_multi_output_parallel_mmul(A, B, C, N, start_col, end_col); });
      start_col += n_cols;
    }

    // Wait for all threads to complete
    for (auto& t : threads)
      t.join();

    // Clear the threads each iteration of the benchmark
    threads.clear();
  }

  // Free memory
  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}
#endif
BENCHMARK(parallel_blocked_column_multi_output_mmul_bench)
  ->Arg(1 * BENCH_SCALE * numThreads * 16)
  ->Arg(2 * BENCH_SCALE * numThreads * 16)
  ->Arg(3 * BENCH_SCALE * numThreads * 16)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

int main(int argc, char** argv)
{
  // Separate user arguments and benchmark arguments
  std::vector<char*> benchmark_args;
  for (int i = 0; i < argc; ++i)
  {
    if (std::string(argv[i]).find("--") == 0 || i == 0)
    {
      // Keep benchmark-specific arguments (starting with '--') and the program name
      benchmark_args.push_back(argv[i]);
    }
    else
    {
      // Custom user arguments
      numThreads = std::min(
        static_cast<unsigned int>(std::stoi(argv[i])), std::thread::hardware_concurrency());
    }
  }
  // Pass filtered arguments to Google Benchmark
  int benchmark_argc = static_cast<int>(benchmark_args.size());
  char** benchmark_argv = benchmark_args.data();

  benchmark::Initialize(&benchmark_argc, benchmark_argv);
  if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
