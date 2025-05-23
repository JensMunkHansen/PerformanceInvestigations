// Main benchmark function of MMul
// By: Nick from CoffeeBeforeArch

#include "benchmark/benchmark.h"

#include "../platform.h"

#include <algorithm>
#include <cstdlib>
#include <immintrin.h>
#include <random>
#include <thread>
#include <vector>

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

// Serial implementation
void serial_avx2_mmul(const float* A, const float* B, float* C, std::size_t N)
{
    // For each row...
    for (std::size_t row = 0; row < N; ++row)
    {
        // For each element in the row/col pair...
        for (std::size_t idx = 0; idx < N; ++idx)
        {
            __m256 a_val = _mm256_set1_ps(A[row * N + idx]);

            for (std::size_t col = 0; col < N; col += 8)
            {
                // Load B and C (aligned)
                __m256 b_vec = _mm256_load_ps(&B[idx * N + col]);
                __m256 c_vec = _mm256_load_ps(&C[row * N + col]);

                // Accumulate the partial results
                c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);

                _mm256_store_ps(&C[row * N + col], c_vec);
            }
        }
    }
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

static void serial_avx2_mmul_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
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
        serial_mmul(A, B, C, N);
    }

    // Free memory
    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
}
BENCHMARK(serial_avx2_mmul_bench)
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
