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

#include <cmath>
#include <iostream>

// Blocked column parallel implementation w/o atomic
void blocked_column_multi_output_parallel_mmul(const float* A, const float* B, float* C,
  std::size_t N, std::size_t start_col, std::size_t end_col)
{
  // For each chunk of columns
#pragma loop(ivdep)
  for (std::size_t col_chunk = start_col; col_chunk < end_col; col_chunk += 16)
  // For each chunk of rows
#pragma loop(ivdep)
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
    // For each block of elements in this row of this column chunk
    // Solve for 16 elements at a time
#pragma loop(ivdep)
      for (std::size_t tile = 0; tile < N; tile += 16)
      // For apply that tile to each row of the row chunk
#pragma loop(ivdep)
        for (std::size_t row = 0; row < 16; row++)
        // For each row in the tile
#pragma loop(ivdep)
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
          // Solve for each element in this tile row
#pragma loop(ivdep)
            for (std::size_t idx = 0; idx < 16; idx++)
              C[(row + row_chunk) * N + col_chunk + idx] +=
                A[(row + row_chunk) * N + tile + tile_row] *
                B[tile * N + tile_row * N + col_chunk + idx];
}

bool matrices_are_close(const float* ref, const float* test, std::size_t N, float eps = 1e-3f)
{
  for (std::size_t i = 0; i < N * N; ++i)
  {
    float a = ref[i], b = test[i];
    float diff = std::fabs(a - b);
    float denom = std::max(1.0f, std::fabs(a));
    if (diff / denom > eps)
    {
      std::cerr << "Mismatch at index " << i << ": " << a << " vs " << b << "\n";
      return false;
    }
  }
  return true;
}

unsigned int numThreads = std::thread::hardware_concurrency();

// Blocked column multi-output serial implementation
void blocked_column_multi_output_mmul_accumulate(
  const float* A, const float* B, float* C, std::size_t N)
{
  constexpr std::size_t block = 16;

  // Assumes N % block == 0
#pragma loop(ivdep)
  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += block)
  {
#pragma loop(ivdep)
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += block)
    {
#pragma loop(ivdep)
      for (std::size_t tile = 0; tile < N; tile += block)
      {
#pragma loop(ivdep)
        for (std::size_t row = 0; row < block; ++row)
        {
          float acc[block] = { 0 };

#pragma loop(ivdep)
          for (std::size_t tile_row = 0; tile_row < block; ++tile_row)
          {
            float A_val = A[(row + row_chunk) * N + tile + tile_row];

#pragma loop(ivdep)
            for (std::size_t idx = 0; idx < block; ++idx)
            {
              acc[idx] += A_val * B[(tile + tile_row) * N + col_chunk + idx];
            }
          }

#pragma loop(ivdep)
          for (std::size_t idx = 0; idx < block; ++idx)
          {
            C[(row + row_chunk) * N + col_chunk + idx] += acc[idx];
          }
        }
      }
    }
  }
}

void blocked_column_multi_output_parallel_mmul_accumulate(const float* A, const float* B, float* C,
  std::size_t N, std::size_t start_col, std::size_t end_col)
{
  const std::size_t block = 16;

  for (std::size_t col_chunk = start_col; col_chunk < end_col; col_chunk += block)
  {
    std::size_t col_limit = std::min(block, N - col_chunk);

    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += block)
    {
      std::size_t row_limit = std::min(block, N - row_chunk);

      for (std::size_t tile = 0; tile < N; tile += block)
      {
        std::size_t tile_limit = std::min(block, N - tile);

        for (std::size_t row = 0; row < row_limit; ++row)
        {
          float acc[16] = { 0 }; // Still capped at 16

          for (std::size_t tile_row = 0; tile_row < tile_limit; ++tile_row)
          {
            float A_val = A[(row + row_chunk) * N + tile + tile_row];
            for (std::size_t idx = 0; idx < col_limit; ++idx)
            {
              acc[idx] += A_val * B[(tile + tile_row) * N + col_chunk + idx];
            }
          }

          for (std::size_t idx = 0; idx < col_limit; ++idx)
          {
            C[(row + row_chunk) * N + col_chunk + idx] += acc[idx];
          }
        }
      }
    }
  }
}

void blocked_column_multi_output_parallel_rows(const float* A, const float* B, float* C,
  std::size_t N, std::size_t start_row, std::size_t end_row)
{
  const std::size_t block = 16;

  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += block)
  {
    std::size_t col_limit = std::min(block, N - col_chunk);

    for (std::size_t row_chunk = start_row; row_chunk < end_row; row_chunk += block)
    {
      std::size_t row_limit = std::min(block, end_row - row_chunk);

      for (std::size_t tile = 0; tile < N; tile += block)
      {
        std::size_t tile_limit = std::min(block, N - tile);

        for (std::size_t row = 0; row < row_limit; ++row)
        {
          float acc[16] = { 0 };

          for (std::size_t tile_row = 0; tile_row < tile_limit; ++tile_row)
          {
            float A_val = A[(row + row_chunk) * N + tile + tile_row];
            for (std::size_t idx = 0; idx < col_limit; ++idx)
            {
              acc[idx] += A_val * B[(tile + tile_row) * N + col_chunk + idx];
            }
          }

          for (std::size_t idx = 0; idx < col_limit; ++idx)
          {
            C[(row + row_chunk) * N + col_chunk + idx] += acc[idx];
          }
        }
      }
    }
  }
}

static void parallel_blocked_row_multi_output_rows_bench(benchmark::State& s)
{
  // Matrix dimensions
  std::size_t N = s.range(0);

  // Random generators
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::uniform_real_distribution<float> dist(-10, 10);

  // Allocate aligned input/output matrices
  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  // Initialize with random values
  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::generate(C, C + N * N, [&] { return 0.0f; });

  // Threads
  std::vector<std::thread> threads;
  threads.reserve(numThreads);

  std::size_t rows_per_thread = (N + numThreads - 1) / numThreads;

  for (auto _ : s)
  {
    std::fill(C, C + N * N, 0.0f); // Ensure clean result for each run

    threads.clear();

    for (std::size_t i = 0; i < numThreads; ++i)
    {
      std::size_t start_row = i * rows_per_thread;
      std::size_t end_row = std::min(N, start_row + rows_per_thread);

      threads.emplace_back(
        [=] { blocked_column_multi_output_parallel_rows(A, B, C, N, start_row, end_row); });
    }

    for (auto& t : threads)
      t.join();
  }

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C);
}
BENCHMARK(parallel_blocked_row_multi_output_rows_bench)
  ->Arg(1 * BENCH_SCALE * numThreads * 16)
  ->Arg(2 * BENCH_SCALE * numThreads * 16)
  ->Arg(3 * BENCH_SCALE * numThreads * 16)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();

// Blocked column multi-output serial implementation
void blocked_column_multi_output_mmul(const float* A, const float* B, float* C, std::size_t N)
{
  // No effect
#pragma clang loop vectorize(enable)
#pragma clang loop interleave(enable)
#pragma clang loop unroll(enable)

  // For each chunk of columns
  for (std::size_t col_chunk = 0; col_chunk < N; col_chunk += 16)
    // For each chunk of rows
    for (std::size_t row_chunk = 0; row_chunk < N; row_chunk += 16)
      // For each block of elements in this row of this column chunk
      // Solve for 16 elements at a time
      for (std::size_t tile = 0; tile < N; tile += 16)
        // Apply that tile to each row of the row chunk
        for (std::size_t row = 0; row < 16; row++)
          // For each row in the tile
          for (std::size_t tile_row = 0; tile_row < 16; tile_row++)
            // Solve for each element in this tile row
            for (std::size_t idx = 0; idx < 16; idx++)
              C[(row + row_chunk) * N + col_chunk + idx] +=
                A[(row + row_chunk) * N + tile + tile_row] *
                B[tile * N + tile_row * N + col_chunk + idx];
}

bool run_parallel_row_correctness_check(std::size_t N)
{
  std::cout << "Running PARALLEL (row-based) correctness check with N = " << N << "...\n";

  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C_ref = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C_test = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  std::mt19937 rng(456); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-10, 10);

  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::fill(C_ref, C_ref + N * N, 0.0f);
  std::fill(C_test, C_test + N * N, 0.0f);

  // Run serial reference
  blocked_column_multi_output_mmul(A, B, C_ref, N);

  // Run row-partitioned parallel version
  std::size_t rows_per_thread = (N + numThreads - 1) / numThreads;
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < numThreads; ++i)
  {
    std::size_t start_row = i * rows_per_thread;
    std::size_t end_row = std::min(N, start_row + rows_per_thread);

    threads.emplace_back(
      [=] { blocked_column_multi_output_parallel_rows(A, B, C_test, N, start_row, end_row); });
  }

  for (auto& t : threads)
    t.join();

  bool ok = matrices_are_close(C_ref, C_test, N);
  std::cout << (ok ? "✅ Row-based parallel version matches reference.\n"
                   : "❌ Row-based parallel version mismatch!\n");

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C_ref);
  ALIGNED_FREE(C_test);
  return ok;
}

bool run_correctness_check(std::size_t N)
{
  std::cout << "Running correctness check with N = " << N << "...\n";

  float* A = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* B = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C_ref = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));
  float* C_test = static_cast<float*>(ALIGNED_ALLOC(64, N * N * sizeof(float)));

  std::mt19937 rng(42); // fixed seed for deterministic results
  std::uniform_real_distribution<float> dist(-10, 10);

  std::generate(A, A + N * N, [&] { return dist(rng); });
  std::generate(B, B + N * N, [&] { return dist(rng); });
  std::fill(C_ref, C_ref + N * N, 0.0f);
  std::fill(C_test, C_test + N * N, 0.0f);

  // Run reference
  blocked_column_multi_output_mmul(A, B, C_ref, N);

  std::size_t num_threads = numThreads;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Calculate values to pass to threads
  // Assumed to be divisable by num_threads (evenly)
  std::size_t n_cols = N / num_threads;

  // Launch threads
  std::size_t start_col = 0;
  for (std::size_t i = 0; i < num_threads; i++)
  {
    auto end_col = start_col + n_cols;
    threads.emplace_back(
      [&] {
        blocked_column_multi_output_parallel_mmul_accumulate(A, B, C_test, N, start_col, end_col);
      });
    start_col += n_cols;
  }

  // Wait for all threads to complete
  for (auto& t : threads)
    t.join();

  // Clear the threads each iteration of the benchmark
  threads.clear();

  bool ok = matrices_are_close(C_ref, C_test, N);
  std::cout << (ok ? "✅ Matrices match within error tolerance.\n" : "❌ Matrices do NOT match.\n");

  ALIGNED_FREE(A);
  ALIGNED_FREE(B);
  ALIGNED_FREE(C_ref);
  ALIGNED_FREE(C_test);
  return ok;
}

int main(int argc, char** argv)
{
  std::vector<char*> benchmark_args;
  for (int i = 0; i < argc; ++i)
  {
    if (std::string(argv[i]).find("--") == 0 || i == 0)
    {
      benchmark_args.push_back(argv[i]);
    }
    else
    {
      numThreads = std::min(
        static_cast<unsigned int>(std::stoi(argv[i])), std::thread::hardware_concurrency());
    }
  }
  if (!run_parallel_row_correctness_check(512))
  {
    std::cerr << "Correctness test failed! Aborting benchmarks.\n";
    return 1;
  }

  int benchmark_argc = static_cast<int>(benchmark_args.size());
  char** benchmark_argv = benchmark_args.data();

  benchmark::Initialize(&benchmark_argc, benchmark_argv);
  if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
