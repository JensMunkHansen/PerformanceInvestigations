// A DMMul reference implementation using MKL

#include <algorithm>
#include <random>
#include <thread>

#include "benchmark/benchmark.h"
#include "mkl.h"

unsigned int numThreads = std::thread::hardware_concurrency();

// Blocked MMul benchmark
static void blocked_aligned_mmul_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    std::size_t N = s.range(0);

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    // Create input matrices
    float* A = (float*)mkl_malloc(N * N * sizeof(float), 64);
    float* B = (float*)mkl_malloc(N * N * sizeof(float), 64);
    float* C = (float*)mkl_malloc(N * N * sizeof(float), 64);

    // MMul scaling constants
    float alpha = 1.0;
    float beta = 0.0;

    // Initialize them with random values (and C to 0)
    std::generate(A, A + N * N, [&] { return dist(rng); });
    std::generate(B, B + N * N, [&] { return dist(rng); });
    std::generate(C, C + N * N, [&] { return 0; });

    // Main benchmark loop
    for (auto _ : s)
    {
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
    }

    // Free memory
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
}
BENCHMARK(blocked_aligned_mmul_bench)
  ->Arg(2 * numThreads * 16)
  ->Arg(4 * numThreads * 16)
  ->Arg(6 * numThreads * 16)
  ->Unit(benchmark::kMillisecond);

// Blocked MMul benchmark
static void blocked_mmul_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    std::size_t N = s.range(0);

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    // Create input matrices
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    // MMul scaling constants
    float alpha = 1.0;
    float beta = 0.0;

    // Initialize them with random values (and C to 0)
    std::generate(A, A + N * N, [&] { return dist(rng); });
    std::generate(B, B + N * N, [&] { return dist(rng); });
    std::generate(C, C + N * N, [&] { return 0; });

    // Main benchmark loop
    for (auto _ : s)
    {
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);
    }

    // Free memory
    free(A);
    free(B);
    free(C);
}
BENCHMARK(blocked_mmul_bench)
  ->Arg(2 * numThreads * 16)
  ->Arg(4 * numThreads * 16)
  ->Arg(6 * numThreads * 16)
  ->Unit(benchmark::kMillisecond);

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
