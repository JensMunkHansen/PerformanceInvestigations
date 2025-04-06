// Main benchmark function of MMul

#include "benchmark/benchmark.h"
#include "../platform.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

unsigned int numThreads = std::thread::hardware_concurrency();

// Blocked serial implementation
void blocked_mmul(const float* A, const float* B, float* C, std::size_t N)
{
  // For each row...
  for (std::size_t row = 0; row < N; row++)
    // For each block in the row...
    // Solve for 16 elements at a time
    for (std::size_t block = 0; block < N; block += 16)
      // For each chunk of A/B for this block
      for (std::size_t chunk = 0; chunk < N; chunk += 16)
        // For each row in the chunk
        for (std::size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++)
          // Go through all the elements in the sub chunk
          for (std::size_t idx = 0; idx < 16; idx++)
            C[row * N + block + idx] +=
              A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + block + idx];
}

// Blocked parallel implementation
void blocked_parallel_mmul(const float* A, const float* B, float* C, std::size_t N,
  std::size_t start_row, std::size_t end_row)
{
  // For each row...
  for (std::size_t row = start_row; row < end_row; row++)
    // For each block in the row...
    // Solve for 16 elements at a time
    for (std::size_t block = 0; block < N; block += 16)
      // For each chunk of A/B for this block
      for (std::size_t chunk = 0; chunk < N; chunk += 16)
        // For each row in the chunk
        for (std::size_t sub_chunk = 0; sub_chunk < 16; sub_chunk++)
          // Go through all the elements in the sub chunk
          for (std::size_t idx = 0; idx < 16; idx++)
            C[row * N + block + idx] +=
              A[row * N + chunk + sub_chunk] * B[chunk * N + sub_chunk * N + block + idx];
}

// Blocked MMul benchmark
static void blocked_mmul_bench(benchmark::State& s)
{
  // Dimensions of our matrix
  std::size_t N = s.range(0);

  // Create our random number generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Create input matrices
  float* A = new float[N * N];
  float* B = new float[N * N];
  float* C = new float[N * N];

  // Initialize them with random values (and C to 0)
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  // Main benchmark loop
  for (auto _ : s)
  {
    blocked_mmul(A, B, C, N);
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(blocked_mmul_bench)
  ->Arg(1 * BENCH_SCALE * numThreads * 16)
  ->Arg(2 * BENCH_SCALE * numThreads * 16)
  ->Arg(3 * BENCH_SCALE * numThreads * 16)
  ->Unit(benchmark::kMillisecond);

// Blocked MMul with aligned memory benchmark
static void blocked_aligned_mmul_bench(benchmark::State& s)
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
    blocked_mmul(A, B, C, N);
  }

  // Free memory, std::free
  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}
BENCHMARK(blocked_aligned_mmul_bench)
  ->Arg(1 * BENCH_SCALE * numThreads * 16)
  ->Arg(2 * BENCH_SCALE * numThreads * 16)
  ->Arg(3 * BENCH_SCALE * numThreads * 16)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Parallel blocked MMul benchmark
static void parallel_blocked_mmul_bench(benchmark::State& s)
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
  std::size_t n_rows = N / num_threads;

  // Main benchmark loop
  for (auto _ : s)
  {
    // Launch threads
    std::size_t end_row = 0;
    for (std::size_t i = 0; i < num_threads; i++)
    {
      auto start_row = i * n_rows;
      end_row = start_row + n_rows;
      threads.emplace_back([&] { blocked_parallel_mmul(A, B, C, N, start_row, end_row); });
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
BENCHMARK(parallel_blocked_mmul_bench)
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
