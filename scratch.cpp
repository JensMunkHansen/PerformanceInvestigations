// Main benchmark function of MMul
// By: Nick from CoffeeBeforeArch

#include "benchmark/benchmark.h"

#include "../platform.h"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#ifndef likely
//#define likely(x) __builtin_expect(!!(x), 1)
#define likely(x) 1
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

unsigned int numThreads = std::thread::hardware_concurrency();

// Serial implementation
void serial_mmul(const float* A, const float* B, float* C, std::size_t N)
{
  // For each row...
  for (std::size_t row = 0; row < N; row++)
    // For each col...
    for (std::size_t col = 0; col < N; col++)
      // For each element in the row/col pair...
      for (std::size_t idx = 0; idx < N; idx++)
        // Accumulate the partial results
        C[row * N + col] += A[row * N + idx] * B[idx * N + col];
}

// Parallel implementation
void parallel_mmul(const float* A, const float* B, float* C, std::size_t N, std::size_t start_row,
  std::size_t end_row)
{
  // For each row assigned to this thread...
  for (std::size_t row = start_row; row < end_row; row++)
    // For each column...
    for (std::size_t col = 0; col < N; col++)
      // For each element in the row-col pair...
      for (std::size_t idx = 0; idx < N; idx++)
        // Accumulate the partial results
        C[row * N + col] += A[row * N + idx] * B[idx * N + col];
}

// Serial MMul benchmark
static void serial_mmul_bench(benchmark::State& s)
{
  // Number Dimensions of our matrix
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
    serial_mmul(A, B, C, N);
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(serial_mmul_bench)
  ->Arg(1 * BENCH_SCALE * 16 * numThreads)
  ->Arg(2 * BENCH_SCALE * 16 * numThreads)
  ->Arg(3 * BENCH_SCALE * 16 * numThreads)
  ->Unit(benchmark::kMillisecond);

// Parallel MMul benchmark
static void parallel_mmul_bench(benchmark::State& s)
{
  // Number Dimensions of our matrix
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
    for (std::size_t i = 0; i < num_threads - 1; i++)
    {
      auto start_row = i * n_rows;
      end_row = start_row + n_rows;
      threads.emplace_back([&] { parallel_mmul(A, B, C, N, start_row, end_row); });
    }

    // Wait for all threads to complete
    for (auto& t : threads)
      t.join();

    // Clear the threads each iteration of the benchmark
    threads.clear();
  }

  // Free memory
  delete[] A;
  delete[] B;
  delete[] C;
}
BENCHMARK(parallel_mmul_bench)
  ->Arg(1 * BENCH_SCALE * 16 * numThreads)
  ->Arg(2 * BENCH_SCALE * 16 * numThreads)
  ->Arg(3 * BENCH_SCALE * 16 * numThreads)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

static void tiled_blocked_parallel_mmul_bench(benchmark::State& s)
{
  const std::size_t N = s.range(0);

  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  const std::size_t num_threads = numThreads;
  const std::size_t tile_size = 128;
  const std::size_t block_size = 16;

  std::size_t num_tiles = (N + tile_size - 1) / tile_size;
  std::vector<std::pair<std::size_t, std::size_t>> tiles;
  for (std::size_t row = 0; row < num_tiles; ++row)
    for (std::size_t col = 0; col < num_tiles; ++col)
      tiles.emplace_back(row, col);

  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);
  for (std::size_t i = 0; i < tiles.size(); ++i)
    thread_tiles[i % num_threads].push_back(tiles[i]);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (auto _ : s)
  {
    std::fill(C, C + N * N, 0.0f);

    for (std::size_t t = 0; t < num_threads; ++t)
    {
      threads.emplace_back(
        [=]
        {
          for (const auto& [tile_row, tile_col] : thread_tiles[t])
          {
            const std::size_t row_start = tile_row * tile_size;
            const std::size_t col_start = tile_col * tile_size;
            const std::size_t row_end = std::min(row_start + tile_size, N);
            const std::size_t col_end = std::min(col_start + tile_size, N);

            for (std::size_t k = 0; k < N; k += tile_size)
            {
              const std::size_t k_end = std::min(k + tile_size, N);

              for (std::size_t i = row_start; i < row_end; i += block_size)
              {
                for (std::size_t j = col_start; j < col_end; j += block_size)
                {
                  float c[block_size][block_size] = { 0 };

                  for (std::size_t kk = k; kk < k_end; ++kk)
                  {
                    float a[block_size];
                  // Prefetch upcoming A and B
#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                    {
                      if (likely(i + bi < N))
                        a[bi] = A[(i + bi) * N + kk];
                      else
                        a[bi] = 0.0f;
                    }

#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bj = 0; bj < block_size; ++bj)
                    {
                      if (likely(j + bj < N))
                      {
                        float b = B[kk * N + (j + bj)];

#pragma ivdep
#pragma clang loop vectorize(enable)
                        for (std::size_t bi = 0; bi < block_size; ++bi)
                        {
                          if (likely(i + bi < N))
                            c[bi][bj] += a[bi] * b;
                        }
                      }
                    }
                  }

#pragma ivdep
#pragma clang loop vectorize(enable)
                  for (std::size_t bi = 0; bi < block_size; ++bi)
                  {
#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bj = 0; bj < block_size; ++bj)
                    {
                      if (likely((i + bi) < N && (j + bj) < N))
                        C[(i + bi) * N + (j + bj)] += c[bi][bj];
                    }
                  }
                }
              }
            }
          }
        });
    }

    for (auto& thread : threads)
      thread.join();
    threads.clear();
  }

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}

BENCHMARK(tiled_blocked_parallel_mmul_bench)
  ->Arg(1 * BENCH_SCALE * 16 * numThreads)
  ->Arg(2 * BENCH_SCALE * 16 * numThreads)
  ->Arg(3 * BENCH_SCALE * 16 * numThreads)
  ->Unit(benchmark::kMillisecond);

static void tiled_blocked_parallel_mmul_perfect_bench(benchmark::State& s)
{
  const std::size_t N = s.range(0);

  // Create random input matrices
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  const std::size_t num_threads = numThreads;
  const std::size_t tile_size = 128; // Must divide N exactly
  const std::size_t block_size = 16; // Must divide tile_size exactly

  // Precompute tile ownership
  std::size_t num_tiles = N / tile_size;
  std::vector<std::pair<std::size_t, std::size_t>> tiles;
  for (std::size_t row = 0; row < num_tiles; ++row)
    for (std::size_t col = 0; col < num_tiles; ++col)
      tiles.emplace_back(row, col);

  // Distribute tiles across threads
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);
  for (std::size_t i = 0; i < tiles.size(); ++i)
    thread_tiles[i % num_threads].push_back(tiles[i]);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (auto _ : s)
  {
    std::fill(C, C + N * N, 0.0f);

    for (std::size_t t = 0; t < num_threads; ++t)
    {
      threads.emplace_back(
        [=]
        {
          for (const auto& [tile_row, tile_col] : thread_tiles[t])
          {
            const std::size_t row_start = tile_row * tile_size;
            const std::size_t col_start = tile_col * tile_size;
            const std::size_t row_end = row_start + tile_size;
            const std::size_t col_end = col_start + tile_size;

            for (std::size_t k = 0; k < N; k += tile_size)
            {
              const std::size_t k_end = std::min(k + tile_size, N);

              for (std::size_t i = row_start; i < row_end; i += block_size)
              {
                for (std::size_t j = col_start; j < col_end; j += block_size)
                {
                  float c[block_size][block_size] = { 0 };

                  for (std::size_t kk = k; kk < k_end; ++kk)
                  {
                    float a[block_size];
#if 0
                    // Prefetch upcoming A and B
                    if (kk + 8 < k_end) // Don't read out of bounds
                    {
                      __builtin_prefetch(&A[(i + 0) * N + (kk + 8)], 0, 3);
                      __builtin_prefetch(&B[(kk + 8) * N + (j + 0)], 0, 3);
                    }
#endif
#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                    {
                      a[bi] = A[(i + bi) * N + kk];
                    }

#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bj = 0; bj < block_size; ++bj)
                    {
                      float b = B[kk * N + (j + bj)];

#pragma ivdep
#pragma clang loop vectorize(enable)
                      for (std::size_t bi = 0; bi < block_size; ++bi)
                      {
                        c[bi][bj] += a[bi] * b;
                      }
                    }
                  }

#pragma ivdep
#pragma clang loop vectorize(enable)
                  for (std::size_t bi = 0; bi < block_size; ++bi)
                  {
#pragma ivdep
#pragma clang loop vectorize(enable)
                    for (std::size_t bj = 0; bj < block_size; ++bj)
                    {
                      C[(i + bi) * N + (j + bj)] += c[bi][bj];
                    }
                  }
                }
              }
            }
          }
        });
    }

    for (auto& thread : threads)
      thread.join();
    threads.clear();
  }

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}

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

/*
//  clang-format off

## Threads and Data Access in Tiled Parallel Matrix Multiply

| Aspect                        | Old naive version                     | New tiled-blocked version              |
|:-------------------------------|:--------------------------------------|:---------------------------------------|
| Thread work assignment         | Rows or columns                      | Disjoint tiles (blocks)                |
| Reads from A                   | Threads read nearly all of A         | Threads read only necessary rows of A  |
| Reads from B                   | Threads read nearly all of B         | Threads read only necessary columns of B |
| Writes to C                    | Threads might write to same regions  | Threads write to disjoint regions (no overlap) |
| Cache conflicts                | Heavy (false sharing, thrashing)     | Minimal (localized per-thread access) |
| Memory bandwidth usage         | Very inefficient                     | Efficient (better cache reuse)         |
| Thread independence            | Poor (data sharing, cache line ping-pong) | Strong (only local data needed)   |
| Scaling with thread count      | Bad after a few cores                | Scales well until memory bandwidth saturation |
| Overlap of A/B data between threads | Large (global)                    | Small (some rows of A shared, acceptable) |
| Prefetching benefit            | Harder, unpredictable                | Easier, structured access patterns     |
| Overall performance            | Poor at high core counts             | Much better and predictable            |
clang on


//  clang-format on
*/
