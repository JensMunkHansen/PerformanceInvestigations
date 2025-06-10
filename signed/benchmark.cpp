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

void compute_scalar_soa(const float* x, const float* y, float* distances, const size_t N)
{
    const float A = -0.5f;
    const float B = 1.0f;
    const float C = -10.0f;
    float rdenom = 1.0f / std::sqrt(A * A + B * B);

    PRAGMA_IVDEP
    for (size_t i = 0; i < N; ++i)
    {
        float xi = x[i];
        float yi = y[i];

        float dist = (A * xi + B * yi + C) * rdenom;
        float yline = (-A * xi - C) / B;

        float sign = (yi < yline) ? -1.0f : 1.0f;
        distances[i] = dist * sign;
    }
}

static void compute_scalar_soa_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    size_t N = 1'000'000;
    const size_t alignment = 64;
    if ((N * sizeof(float)) % alignment != 0)
        N += (alignment / sizeof(float)) - (N % (alignment / sizeof(float)));

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    float* y = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));
    float* x = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));
    float* distances = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));

    std::generate(x, x + N, [&] { return dist(rng); });
    std::generate(y, y + N, [&] { return dist(rng); });

    const int innerLoops = 100; // Repeat multiple times per iteration
    // Main benchmark loop
    for (auto _ : s)
    {
        for (int i = 0; i < innerLoops; ++i)
        {
            compute_scalar_soa(x, y, distances, N);
        }
    }
    s.SetItemsProcessed(s.iterations() * innerLoops * N);

    // Free memory
    ALIGNED_FREE(y);
    ALIGNED_FREE(x);
    ALIGNED_FREE(distances);
}
BENCHMARK(compute_scalar_soa_bench)
  ->Iterations(1000000)
  ->Unit(benchmark::kNanosecond)
  ->UseRealTime();

#if 0
    // Optional: One Newton-Raphson iteration for better precision
    // denom_rsqrt = 0.5 * denom_rsqrt * (3 - x * rsqrt^2)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);
    __m256 denom_sq = _mm256_set1_ps(denom_scalar * denom_scalar);
    __m256 tmp = _mm256_mul_ps(denom_rsqrt, denom_rsqrt);
    tmp = _mm256_mul_ps(tmp, denom_sq);
    tmp = _mm256_sub_ps(three, tmp);
    tmp = _mm256_mul_ps(tmp, half);

#endif

void compute_avx2_soa(const float* x, const float* y, float* distances, size_t N)
{
    assert(x && y && distances);
    const float A = -0.5f;
    const float B = 1.0f;
    const float C = -10.0f;
    size_t i = 0;
    const size_t stride = 8;

    __m256 a_vec = _mm256_set1_ps(A);
    __m256 b_vec = _mm256_set1_ps(B);
    __m256 c_vec = _mm256_set1_ps(C);

    // Use rsqrt approximation with optional refinement
    float denom_scalar = std::sqrt(A * A + B * B);
    __m256 denom_rsqrt = _mm256_rsqrt_ps(_mm256_set1_ps(denom_scalar * denom_scalar));

    for (; i + stride - 1 < N; i += stride)
    {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);

        __m256 ax = _mm256_mul_ps(a_vec, x_vec);
        __m256 by = _mm256_mul_ps(b_vec, y_vec);
        __m256 numerator = _mm256_add_ps(_mm256_add_ps(ax, by), c_vec);
        __m256 dist = _mm256_mul_ps(numerator, denom_rsqrt);

        __m256 neg_ax = _mm256_sub_ps(_mm256_setzero_ps(), ax);
        __m256 yline = _mm256_div_ps(_mm256_sub_ps(neg_ax, c_vec), b_vec);
        __m256 mask = _mm256_cmp_ps(y_vec, yline, _CMP_LT_OS);

        __m256 neg_dist = _mm256_sub_ps(_mm256_setzero_ps(), dist);
        __m256 result = _mm256_blendv_ps(dist, neg_dist, mask);

        _mm256_storeu_ps(&distances[i], result);
    }

    // Scalar tail
    for (; i < N; ++i)
    {
        float xi = x[i];
        float yi = y[i];
        float dist = (A * xi + B * yi + C) / std::sqrt(A * A + B * B);
        float yline = (-A * xi - C) / B;
        float sign = (yi < yline) ? -1.0f : 1.0f;
        distances[i] = dist * sign;
    }
}
static void compute_scalar_avx_soa_bench(benchmark::State& s)
{
    // Number Dimensions of our matrix
    size_t N = 1'000'000;
    const size_t alignment = 64;
    if ((N * sizeof(float)) % alignment != 0)
        N += (alignment / sizeof(float)) - (N % (alignment / sizeof(float)));

    // Create our random number generators
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_real_distribution<float> dist(-10, 10);

    float* y = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));
    float* x = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));
    float* distances = static_cast<float*>(ALIGNED_ALLOC(64, N * sizeof(float)));

    std::generate(x, x + N, [&] { return dist(rng); });
    std::generate(y, y + N, [&] { return dist(rng); });

    // Main benchmark loop
    const int innerLoops = 100; // Repeat multiple times per iteration
    // Main benchmark loop
    for (auto _ : s)
    {
        for (int i = 0; i < innerLoops; ++i)
        {
            compute_avx2_soa(x, y, distances, N);
        }
    }
    s.SetItemsProcessed(s.iterations() * innerLoops * N);
    // Free memory
    ALIGNED_FREE(y);
    ALIGNED_FREE(x);
    ALIGNED_FREE(distances);
}
BENCHMARK(compute_scalar_avx_soa_bench)
  ->Iterations(1000000)
  ->Unit(benchmark::kNanosecond)
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
