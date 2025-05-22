#include "benchmark/benchmark.h"

#include "../platform.h"
#include "../threadpool.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include <immintrin.h>

#include <oneapi/tbb/task_group.h>

#include <oneapi/tbb.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

ThreadPool pool(std::thread::hardware_concurrency());

unsigned int numThreads = std::thread::hardware_concurrency();

// Serial implementation
void serial_mmul(
  const float* A, const float* B, float* C, std::size_t L, std::size_t M, std::size_t N)
{
    // For each row...
    for (std::size_t row = 0; row < L; row++)
        // For each col...
        for (std::size_t col = 0; col < N; col++)
            // For each element in the row/col pair...
            for (std::size_t idx = 0; idx < M; idx++)
                // Accumulate the partial results
                C[row * N + col] += A[row * M + idx] * B[idx * N + col];
}

// Serial MMul benchmark
static void serial_mmul_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    std::size_t M = s.range(0);
    std::size_t L = 16;
    std::size_t N = 16;

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    // Create input matrices
    float* A = new float[L * M];
    float* B = new float[M * N];
    float* C = new float[L * N];

    // Initialize them with random values (and C to 0)
    std::generate(A, A + L * M, [&] { return dist(rng); });
    std::generate(B, B + M * N, [&] { return dist(rng); });
    std::generate(C, C + L * N, [&] { return 0.0f; });

    // Main benchmark loop
    for (auto _ : s)
    {
        serial_mmul(A, B, C, L, M, N);
    }

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
}
BENCHMARK(serial_mmul_bench)
  ->Arg(1 * 8 * 16 * numThreads)
  ->Arg(2 * 8 * 16 * numThreads)
  ->Arg(3 * 8 * 16 * numThreads)
  ->Unit(benchmark::kMillisecond);

/**
 * @brief Compute and accumulate a tile of the output matrix C in C[L×N] = A[L×M] × B[M×N]
 *
 * This function computes a single tile of the output matrix `C` using blocked matrix
 * multiplication. The tile has dimensions `[tile_size × tile_size]`, starting at `row_start` and
 * `col_start`. It accumulates the partial product contribution from a tile of A and B along the
 * shared inner dimension starting at position `k`. This function is called multiple times per
 * output tile to sum over all matching inner-dimension tiles (`k` ranges from 0 to M in steps of
 * tile_size).
 *
 * Internally, the function performs a fine-grained blocking of size `block_size` (typically 16×16),
 * using a local register- or stack-resident accumulator matrix `c[block_size][block_size]` to
 * collect the partial results. These blocks are computed in a layout that favors cache reuse and
 * vectorization.
 *
 * The function handles partial (edge) tiles safely by bounding all memory accesses with `(i + bi) <
 * L` and `(j + bj) < N`, and clipping iteration ranges via `std::min(...)` where appropriate.
 *
 * @param A Pointer to input matrix A, shape [L×M], row-major.
 * @param B Pointer to input matrix B, shape [M×N], row-major.
 * @param C Pointer to output matrix C, shape [L×N], row-major.
 * @param L Number of rows in matrix A and C.
 * @param M Number of columns in A and rows in B.
 * @param N Number of columns in B and C.
 * @param row_start Starting row index in C (and A) for this tile.
 * @param col_start Starting column index in C (and B) for this tile.
 * @param k Starting index along the inner dimension (shared dim) for the partial product.
 * @param tile_size Size of the square tile being computed (typically 128).
 *
 * @note This function assumes that input matrices are aligned and padded appropriately
 *       for SIMD performance. Alignment directives like `alignas(64)` are used for stack-allocated
 * buffers.
 * @note All matrix accesses are bounds-checked using clipped loop ranges and guards to prevent
 * invalid reads/writes.
 * @note This function is safe to use for edge tiles.
 */
template <size_t block_size>
inline void execute_tile_fast(const float* __restrict A, const float* __restrict B,
  float* __restrict C, std::size_t L, std::size_t M, std::size_t N, std::size_t row_start,
  std::size_t col_start, std::size_t k, std::size_t tile_size)
{
    const std::size_t row_end = row_start + tile_size;
    const std::size_t col_end = col_start + tile_size;
    const std::size_t k_end = k + tile_size;

    for (std::size_t i = row_start; i < row_end; i += block_size)
    {
        for (std::size_t j = col_start; j < col_end; j += block_size)
        {
            alignas(64) float c[block_size][block_size] = { 0 };

            for (std::size_t kk = k; kk < k_end; ++kk)
            {
                alignas(64) float a[block_size];
                PRAGMA_IVDEP
                for (std::size_t bi = 0; bi < block_size; ++bi)
                    a[bi] = A[(i + bi) * M + kk];

                PRAGMA_IVDEP
                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    float b = B[kk * N + (j + bj)];
                    PRAGMA_IVDEP
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                        c[bi][bj] += a[bi] * b;
                }
            }

            PRAGMA_IVDEP
            for (std::size_t bi = 0; bi < block_size; ++bi)
            {
                PRAGMA_IVDEP
                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    C[(i + bi) * N + (j + bj)] += c[bi][bj];
                }
            }
        }
    }
}

template <size_t block_size>
inline void execute_tile_edge(const float* A, const float* B, float* C, std::size_t L,
  std::size_t M, std::size_t N, std::size_t row_start, std::size_t col_start, std::size_t k,
  std::size_t tile_size)
{
    const std::size_t row_end = std::min(row_start + tile_size, L);
    const std::size_t col_end = std::min(col_start + tile_size, N);
    const std::size_t k_end = std::min(k + tile_size, M);

    for (std::size_t i = row_start; i < row_end; i += block_size)
    {
        for (std::size_t j = col_start; j < col_end; j += block_size)
        {
            float c[block_size][block_size] = { 0 };

            for (std::size_t kk = k; kk < k_end; ++kk)
            {
                float a[block_size];
                for (std::size_t bi = 0; bi < block_size; ++bi)
                    a[bi] = (i + bi < L) ? A[(i + bi) * M + kk] : 0.0f;

                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    float b = (j + bj < N) ? B[kk * N + (j + bj)] : 0.0f;
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                        if ((i + bi < L) && (j + bj < N))
                            c[bi][bj] += a[bi] * b;
                }
            }

            for (std::size_t bi = 0; bi < block_size; ++bi)
                for (std::size_t bj = 0; bj < block_size; ++bj)
                    if ((i + bi < L) && (j + bj < N))
                        C[(i + bi) * N + (j + bj)] += c[bi][bj];
        }
    }
}

/**
 * @brief Perform a high-performance tiled matrix multiplication: C[L×N] = A[L×M] × B[M×N]
 *
 * This function divides the output matrix C into square tiles of fixed size (e.g. 128×128),
 * and computes each tile as the sum of partial dot products using tiles from A and B.
 *
 * Full tiles (completely contained within matrix bounds) are prioritized for early execution,
 * while edge tiles (that touch matrix borders) are deferred to the end. This improves cache
 * locality and overall performance by front-loading compute-heavy tiles.
 *
 * Each tile is further subdivided into smaller blocks (e.g. 16×16) for SIMD-friendly
 * accumulation within a temporary local tile buffer, reducing memory traffic to the global
 * result matrix C.
 *
 * Tile execution is parallelized using a user-provided thread pool. Each thread receives a
 * stripe of tiles following a diagonal pattern, e.g., thread t starts at tile (t, t), (t+1,
 * t+1), etc.
 *
 * @param A Pointer to matrix A of size [L×M], stored in row-major order.
 * @param B Pointer to matrix B of size [M×N], stored in row-major order.
 * @param C Pointer to output matrix C of size [L×N], stored in row-major order.
 * @param L Number of rows in matrix A and C.
 * @param M Number of columns in A and rows in B (shared inner dimension).
 * @param N Number of columns in B and C.
 * @param num_threads Number of threads to divide tile work across.
 * @param pool ThreadPool instance used to parallelize tile execution.
 *
 * @note All matrices must be aligned and padded appropriately for performance.
 *       Tile size and block size are compile-time constants optimized for target
 * cache/microarchitecture.
 */
template <size_t tile_size>
static void tiled_blocked_parallel_mmul_general(const float* A, const float* B, float* C,
  std::size_t L, std::size_t M, std::size_t N, std::size_t num_threads, ThreadPool& pool)
{
    const std::size_t num_tile_rows = (L + tile_size - 1) / tile_size;
    const std::size_t num_tile_cols = (N + tile_size - 1) / tile_size;

    // Step 1: Split into full (hot) and edge (cold) tiles
    std::vector<std::pair<std::size_t, std::size_t>> full_tiles;
    std::vector<std::pair<std::size_t, std::size_t>> edge_tiles;

    for (std::size_t row = 0; row < num_tile_rows; ++row)
    {
        for (std::size_t col = 0; col < num_tile_cols; ++col)
        {
            bool is_edge = ((row + 1) * tile_size > L) || ((col + 1) * tile_size > N);
            if (is_edge)
                edge_tiles.emplace_back(row, col);
            else
                full_tiles.emplace_back(row, col);
        }
    }

    // Step 2: Assign hot tiles diagonally to threads
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);
    for (std::size_t i = 0; i < full_tiles.size(); ++i)
    {
        std::size_t t = (full_tiles[i].first + full_tiles[i].second) % num_threads;
        thread_tiles[t].push_back(full_tiles[i]);
    }

#if 1

#if 0
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

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      execute_tile_fast<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          });
    }

    // Step 4: Launch cold tiles (fallback kernel)
    const std::size_t edge_tiles_per_thread = (edge_tiles.size() + num_threads - 1) / num_threads;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        std::size_t begin = t * edge_tiles_per_thread;
        std::size_t end = std::min(begin + edge_tiles_per_thread, edge_tiles.size());

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

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      bool edge_k = (k + tile_size > M);
                      if (edge_k)
                      {
                          execute_tile_edge<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                      }
                      else
                      {
                          execute_tile_fast<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                      }
                  }
              }
          });
    }

    // Step 5: Wait for all tasks to complete
    tg.wait();
#else

    tbb::affinity_partitioner ap;

    tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, thread_tiles.size()),
      [&](const tbb::blocked_range<std::size_t>& r)
      {
          for (std::size_t t = r.begin(); t < r.end(); ++t)
          {
              for (const auto& [tile_row, tile_col] : thread_tiles[t])
              {
                  std::size_t row_start = tile_row * tile_size;
                  std::size_t col_start = tile_col * tile_size;

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      execute_tile_fast<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          }
      },
      ap);

    tbb::parallel_for(
      tbb::blocked_range<std::size_t>(0, edge_tiles.size()),
      [&](const tbb::blocked_range<std::size_t>& r)
      {
          for (std::size_t i = r.begin(); i < r.end(); ++i)
          {
              const auto& [tile_row, tile_col] = edge_tiles[i];

              std::size_t row_start = tile_row * tile_size;
              std::size_t col_start = tile_col * tile_size;

              for (std::size_t k = 0; k < M; k += tile_size)
              {
                  bool edge_k = (k + tile_size > M);
                  if (edge_k)
                  {
                      execute_tile_edge<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
                  else
                  {
                      execute_tile_fast<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          }
      },
      ap);

#endif

#else
    std::vector<std::future<void>> futures;

    // Step 3: Launch hot tiles first (high-performance kernel)
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        futures.emplace_back(pool.submit(
          [=]
          {
              for (const auto& [tile_row, tile_col] : thread_tiles[t])
              {
                  std::size_t row_start = tile_row * tile_size;
                  std::size_t col_start = tile_col * tile_size;

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      // Middle tiles are guaranteed safe
                      execute_tile_fast<16>(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          }));
    }

    // Step 4: Launch cold tiles after (safe fallback kernel)
    const std::size_t edge_tiles_per_thread = (edge_tiles.size() + num_threads - 1) / num_threads;
    for (std::size_t t = 0; t < num_threads; ++t)
    {
        std::size_t begin = t * edge_tiles_per_thread;
        std::size_t end = std::min(begin + edge_tiles_per_thread, edge_tiles.size());

        if (begin >= end)
            continue;

        futures.emplace_back(pool.submit(
          [=]
          {
              for (std::size_t i = begin; i < end; ++i)
              {
                  auto [tile_row, tile_col] = edge_tiles[i];
                  std::size_t row_start = tile_row * tile_size;
                  std::size_t col_start = tile_col * tile_size;

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      bool edge_k = (k + tile_size > M);
                      if (edge_k)
                      {
                          execute_tile_edge<16>(
                            A, B, C, L, M, N, row_start, col_start, k, tile_size);
                      }
                      else
                      {
                          execute_tile_fast<16>(
                            A, B, C, L, M, N, row_start, col_start, k, tile_size);
                      }
                  }
              }
          }));
    }

    // Step 5: Wait for all tasks
    for (auto& fut : futures)
        fut.get();
#endif
}

static void tiled_blocked_parallel_mmul_bench(benchmark::State& s)
{
    const std::size_t M = s.range(0);
    const std::size_t L = 16;
    const std::size_t N = 16;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-10, 10);

    float* A = static_cast<float*>(ALIGNED_ALLOC(64, L * M * sizeof(float)));
    float* B = static_cast<float*>(ALIGNED_ALLOC(64, M * N * sizeof(float)));
    float* C = static_cast<float*>(ALIGNED_ALLOC(64, L * N * sizeof(float)));

    std::generate(A, A + L * M, [&] { return dist(rng); });
    std::generate(B, B + M * N, [&] { return dist(rng); });

    for (auto _ : s)
    {
        std::fill(C, C + L * N, 0.0f);
        tiled_blocked_parallel_mmul_general<16>(A, B, C, L, M, N, numThreads, pool);
    }

    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
}

BENCHMARK(tiled_blocked_parallel_mmul_bench)
  ->Arg(1 * 8 * 16 * numThreads)
  ->Arg(2 * 8 * 16 * numThreads)
  ->Arg(3 * 8 * 16 * numThreads)
  ->Unit(benchmark::kMillisecond);

static void serial_subscan_mmul_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    std::size_t M = 128;
    std::size_t L = 128;
    std::size_t N = 128;

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    // Create input matrices
    float* A = static_cast<float*>(ALIGNED_ALLOC(64, L * M * sizeof(float)));
    float* B = static_cast<float*>(ALIGNED_ALLOC(64, M * N * sizeof(float)));
    float* C = static_cast<float*>(ALIGNED_ALLOC(64, L * N * sizeof(float)));

    // Initialize them with random values (and C to 0)
    std::generate(A, A + L * M, [&] { return dist(rng); });
    std::generate(B, B + M * N, [&] { return dist(rng); });
    std::generate(C, C + L * N, [&] { return 0.0f; });

    // Main benchmark loop
    for (auto _ : s)
    {
        serial_mmul(A, B, C, L, M, N);
    }

    // Free memory
    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
}
BENCHMARK(serial_subscan_mmul_bench)->Arg(1)->Unit(benchmark::kMillisecond);

static void tiled_subscan_parallel_mmul_bench(benchmark::State& s)
{
    const std::size_t M = 128;
    const std::size_t L = 128;
    const std::size_t N = 128;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-10, 10);

    float* A = static_cast<float*>(ALIGNED_ALLOC(64, L * M * sizeof(float)));
    float* B = static_cast<float*>(ALIGNED_ALLOC(64, M * N * sizeof(float)));
    float* C = static_cast<float*>(ALIGNED_ALLOC(64, L * N * sizeof(float)));

    std::generate(A, A + L * M, [&] { return dist(rng); });
    std::generate(B, B + M * N, [&] { return dist(rng); });

    for (auto _ : s)
    {
        std::fill(C, C + L * N, 0.0f);
        tiled_blocked_parallel_mmul_general<16>(A, B, C, L, M, N, numThreads, pool);
    }

    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
}
BENCHMARK(tiled_subscan_parallel_mmul_bench)->Arg(1)->Unit(benchmark::kMillisecond);

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

    // Pass filtered arguments to Google Benchmark
    int benchmark_argc = static_cast<int>(benchmark_args.size());
    char** benchmark_argv = benchmark_args.data();

    benchmark::Initialize(&benchmark_argc, benchmark_argv);
    if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
