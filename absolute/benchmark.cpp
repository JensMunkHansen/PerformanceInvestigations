
#include "benchmark/benchmark.h"

#include "../platform.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <immintrin.h>

#define USE_TBB 1

#ifdef USE_TBB
#include <oneapi/tbb.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_group.h>
#else
#include "../threadpool.hpp"
#endif

#ifndef USE_TBB
ThreadPool pool(std::thread::hardware_concurrency());
#endif

unsigned int numThreads = std::thread::hardware_concurrency();

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
// BENCHMARK(serial_mmul_bench)
//   ->Arg(1 * BENCH_SCALE * 16 * numThreads)
//   ->Arg(2 * BENCH_SCALE * 16 * numThreads)
//   ->Arg(3 * BENCH_SCALE * 16 * numThreads)
//   ->Unit(benchmark::kMillisecond);

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
#if 0
BENCHMARK(parallel_mmul_bench)
  ->Arg(1 * BENCH_SCALE * 16 * numThreads)
  ->Arg(2 * BENCH_SCALE * 16 * numThreads)
  ->Arg(3 * BENCH_SCALE * 16 * numThreads)
  ->Unit(benchmark::kMillisecond)
  ->UseRealTime();
#endif

/**
 * @brief Compute partial results and update a tile in C of size [tile_size, tile_size]
 *
 * @param A pointer to input matrix [N,N]
 * @param B pointer to input matrix [N,N]
 * @param C pointer to output matrix [N,N]
 * @param N size parameter
 * @param row_start in A
 * @param col_start in B
 * @param k runing parameter for patch used for update the tile in C
 * @param tile_size [typically 128]
 */

inline void execute_tile_fast(const float* __restrict A, const float* __restrict B,
  float* __restrict C, std::size_t N, std::size_t row_start, std::size_t col_start, std::size_t k,
  std::size_t tile_size)
{
    const std::size_t row_end = row_start + tile_size;
    const std::size_t col_end = col_start + tile_size;
    const std::size_t k_end = k + tile_size;
    const std::size_t block_size = 16;

    for (std::size_t i = row_start; i < row_end; i += block_size)
    {
        for (std::size_t j = col_start; j < col_end; j += block_size)
        {
            // Update local c
            alignas(64) float c[block_size][block_size] = { 0 };

            for (std::size_t kk = k; kk < k_end; ++kk)
            {
                alignas(64) float a[block_size];
                // Without this, we loose a factor of 3 for large arrays.
                PRAGMA_IVDEP
                for (std::size_t bi = 0; bi < block_size; ++bi)
                    a[bi] = A[(i + bi) * N + kk];
                PRAGMA_IVDEP
                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    float b = B[kk * N + (j + bj)];
                    PRAGMA_IVDEP
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                    {
                        c[bi][bj] += a[bi] * b;
                    }
                }
            }
            // Update global C
            PRAGMA_IVDEP
            for (std::size_t bi = 0; bi < block_size; ++bi)
            {
                PRAGMA_IVDEP
                for (std::size_t bj = 0; bj < block_size; ++bj)
                    C[(i + bi) * N + (j + bj)] += c[bi][bj];
            }
        }
    }
}

inline void execute_tile_edge(const float* A, const float* B, float* C, std::size_t N,
  std::size_t row_start, std::size_t col_start, std::size_t k, std::size_t tile_size)
{
    const std::size_t row_end = std::min(row_start + tile_size, N);
    const std::size_t col_end = std::min(col_start + tile_size, N);
    const std::size_t k_end = std::min(k + tile_size, N);
    const std::size_t block_size = 16;

    for (std::size_t i = row_start; i < row_end; i += block_size)
    {
        for (std::size_t j = col_start; j < col_end; j += block_size)
        {
            alignas(64) float c[block_size][block_size] = { 0 };

            for (std::size_t kk = k; kk < k_end; ++kk)
            {
                alignas(64) float a[block_size];
                for (std::size_t bi = 0; bi < block_size; ++bi)
                    a[bi] = (i + bi < N) ? A[(i + bi) * N + kk] : 0.0f;

                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    float b = (j + bj < N) ? B[kk * N + (j + bj)] : 0.0f;
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                        if (i + bi < N && j + bj < N)
                            c[bi][bj] += a[bi] * b;
                }
            }

            for (std::size_t bi = 0; bi < block_size; ++bi)
                for (std::size_t bj = 0; bj < block_size; ++bj)
                    if ((i + bi) < N && (j + bj) < N)
                        C[(i + bi) * N + (j + bj)] += c[bi][bj];
        }
    }
}

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

    std::vector<std::pair<std::size_t, std::size_t>> full_tiles;
    std::vector<std::pair<std::size_t, std::size_t>> edge_tiles;

    for (std::size_t row = 0; row < num_tiles; ++row)
    {
        for (std::size_t col = 0; col < num_tiles; ++col)
        {
            bool is_edge = (row + 1) * tile_size > N || (col + 1) * tile_size > N;
            if (is_edge)
                edge_tiles.emplace_back(row, col);
            else
                full_tiles.emplace_back(row, col);
        }
    }
    // Balanced round-robin assignment of full tiles
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);
    for (std::size_t i = 0; i < full_tiles.size(); ++i)
    {
        thread_tiles[i % num_threads].push_back(full_tiles[i]);
    }
    const size_t num_edge_tiles = edge_tiles.size();
    const std::size_t tiles_per_thread = (num_edge_tiles + num_threads - 1) / num_threads;

    for (auto _ : s)
    {
        std::fill(C, C + N * N, 0.0f);
#ifdef USE_TBB
#if 1
        // Faster since balanced
        tbb::task_group tg;
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            tg.run(
              [=]
              {
                  for (const auto& [tile_row, tile_col] : thread_tiles[t])
                  {
                      std::size_t row_start = tile_row * tile_size;
                      std::size_t col_start = tile_col * tile_size;

                      for (std::size_t k = 0; k < N; k += tile_size)
                      {
                          execute_tile_fast(A, B, C, N, row_start, col_start, k, tile_size);
                      }
                  }
              });
        }

        // Cold tile (boundaries)
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            std::size_t begin = t * tiles_per_thread;
            std::size_t end = std::min(begin + tiles_per_thread, edge_tiles.size());

            if (begin >= end)
                continue;

            tg.run(
              [=]
              {
                  for (std::size_t i = begin; i < end; ++i)
                  {
                      auto [tile_row, tile_col] = edge_tiles[i];
                      std::size_t row_start = tile_row * tile_size;
                      std::size_t col_start = tile_col * tile_size;

                      for (std::size_t k = 0; k < N; k += tile_size)
                      {
                          execute_tile_edge(A, B, C, N, row_start, col_start, k, tile_size);
                      }
                  }
              });
        }
        tg.wait();
#else
        tbb::affinity_partitioner ap;

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, thread_tiles.size()),
          [&](const tbb::blocked_range<std::size_t>& r)
          {
              for (std::size_t t = r.begin(); t < r.end(); ++t)
              {
                  for (const auto& [tile_row, tile_col] : thread_tiles[t])
                  {
                      std::size_t row_start = tile_row * tile_size;
                      std::size_t col_start = tile_col * tile_size;

                      // Loop over tiles for computing a final tile in C
                      for (std::size_t k = 0; k < N; k += tile_size)
                      {
                          execute_tile_fast(A, B, C, N, row_start, col_start, k, tile_size);
                      }
                  }
              }
          });

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, edge_tiles.size()),
          [&](const tbb::blocked_range<std::size_t>& r)
          {
              for (std::size_t t = r.begin(); t < r.end(); ++t)
              {
                  auto [tile_row, tile_col] = edge_tiles[t];

                  std::size_t row_start = tile_row * tile_size;
                  std::size_t col_start = tile_col * tile_size;

                  for (std::size_t k = 0; k < N; k += tile_size)
                  {
                      execute_tile_edge(A, B, C, N, row_start, col_start, k, tile_size);
                  }
              }
          });
#endif
#else
        std::vector<std::future<void>> futures;

        for (std::size_t t = 0; t < num_threads; ++t)
        {
            futures.emplace_back(pool.submit(
              [=]
              {
                  for (const auto& [tile_row, tile_col] : thread_tiles[t])
                  {
                      std::size_t row_start = tile_row * tile_size;
                      std::size_t col_start = tile_col * tile_size;

                      // Loop over tiles for computing a final tile in C
                      for (std::size_t k = 0; k < N; k += tile_size)
                      {
                          execute_tile_fast(A, B, C, N, row_start, col_start, k, tile_size);
                      }
                  }
              }));
        }
        for (std::size_t t = 0; t < num_threads; ++t)
        {
            std::size_t begin = t * tiles_per_thread;
            std::size_t end = std::min(begin + tiles_per_thread, num_edge_tiles);

            futures.emplace_back(pool.submit(
              [=]
              {
                  for (std::size_t i = begin; i < end; ++i)
                  {
                      auto [tile_row, tile_col] = edge_tiles[i];

                      std::size_t row_start = tile_row * tile_size;
                      std::size_t col_start = tile_col * tile_size;

                      for (std::size_t k = 0; k < N; k += tile_size)
                      {
                          execute_tile_edge(A, B, C, N, row_start, col_start, k, tile_size);
                      }
                  }
              }));
        }
        for (auto& fut : futures)
            fut.get();
#endif
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
    serial_mmul(A, B, C_ref, N);

    const std::size_t num_threads = numThreads;
    const std::size_t tile_size = 128; // Must divide N exactly
    const std::size_t block_size = 16; // Must divide tile_size exactly

    std::size_t num_tiles = (N + tile_size - 1) / tile_size;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);

    for (std::size_t thread_id = 0; thread_id < num_threads; ++thread_id)
    {
        if (thread_id >= num_tiles)
            continue; // Extra threads won't be assigned anything

        std::size_t row = thread_id;
        for (std::size_t i = 0; i < num_tiles; ++i)
        {
            std::size_t col = (thread_id + i) % num_tiles; // Wraparound start
            thread_tiles[thread_id].emplace_back(row, col);
        }
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

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
                                  PRAGMA_IVDEP
                                  for (std::size_t bi = 0; bi < block_size; ++bi)
                                  {
                                      a[bi] = A[(i + bi) * N + kk];
                                  }

                                  PRAGMA_IVDEP
                                  for (std::size_t bj = 0; bj < block_size; ++bj)
                                  {
                                      float b = B[kk * N + (j + bj)];

                                      PRAGMA_IVDEP
                                      for (std::size_t bi = 0; bi < block_size; ++bi)
                                      {
                                          c[bi][bj] += a[bi] * b;
                                      }
                                  }
                              }

                              PRAGMA_IVDEP
                              for (std::size_t bi = 0; bi < block_size; ++bi)
                              {
                                  PRAGMA_IVDEP
                                  for (std::size_t bj = 0; bj < block_size; ++bj)
                                  {
                                      C_test[(i + bi) * N + (j + bj)] += c[bi][bj];
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

    bool ok = matrices_are_close(C_ref, C_test, N);
    std::cout << (ok ? "✅ Matrices match within error tolerance.\n" : "❌ Matrices do NOT match.\n");

    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C_test);
    ALIGNED_FREE(C_ref);
    return ok;
}

int main(int argc, char** argv)
{
    // Separate user arguments and benchmark arguments
    std::vector<char*> benchmark_args;
    for (int i = 0; i < argc; ++i)
    {
        if (std::string(argv[i]).find("--") == 0 || i == 0)
        {
            // Keep benchmark-specific arguments (starting with '--') and the
            // program name
            benchmark_args.push_back(argv[i]);
        }
        else
        {
            // Custom user arguments
            numThreads = std::min(
              static_cast<unsigned int>(std::stoi(argv[i])), std::thread::hardware_concurrency());
        }
    }
#if 0
    if (!run_correctness_check(512))
    {
        std::cerr << "Correctness test failed! Aborting benchmarks.\n";
        return 1;
    }
#endif
    // Pass filtered arguments to Google Benchmark
    int benchmark_argc = static_cast<int>(benchmark_args.size());
    char** benchmark_argv = benchmark_args.data();

    benchmark::Initialize(&benchmark_argc, benchmark_argv);
    if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
