#include <cassert>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

// Auto-vectorized scalar version (SoA + restrict + ivdep)
void compute_scalar_soa_restrict(const float* __restrict x, const float* __restrict y,
  float* __restrict distances, size_t N, float A, float B, float C)
{
    float denom = std::sqrt(A * A + B * B);

#pragma GCC ivdep
    for (size_t i = 0; i < N; ++i)
    {
        float xi = x[i];
        float yi = y[i];

        float dist = (A * xi + B * yi + C) / denom;
        float yline = (-A * xi - C) / B;

        float sign = (yi < yline) ? -1.0f : 1.0f;
        distances[i] = dist * sign;
    }
}

// Manual AVX2 implementation
void compute_avx2_soa(const float* __restrict x, const float* __restrict y,
  float* __restrict distances, size_t N, float A, float B, float C)
{
    assert(x && y && distances);
    size_t i = 0;
    const size_t stride = 8;

    __m256 a_vec = _mm256_set1_ps(A);
    __m256 b_vec = _mm256_set1_ps(B);
    __m256 c_vec = _mm256_set1_ps(C);
    __m256 denom = _mm256_set1_ps(std::sqrt(A * A + B * B));

    for (; i + stride - 1 < N; i += stride)
    {
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        __m256 y_vec = _mm256_loadu_ps(&y[i]);

        __m256 ax = _mm256_mul_ps(a_vec, x_vec);
        __m256 by = _mm256_mul_ps(b_vec, y_vec);
        __m256 numerator = _mm256_add_ps(_mm256_add_ps(ax, by), c_vec);
        __m256 dist = _mm256_div_ps(numerator, denom);

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

int main()
{
    constexpr size_t N = 1 << 20; // 1M points
    std::vector<float> x(N), y(N), d_scalar(N), d_avx2(N);

    // Fill with random values
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t i = 0; i < N; ++i)
    {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    float A = -0.5f, B = 1.0f, C = -10.0f;

    // Scalar version
    auto t1 = std::chrono::high_resolution_clock::now();
    compute_scalar_soa_restrict(x.data(), y.data(), d_scalar.data(), N, A, B, C);
    auto t2 = std::chrono::high_resolution_clock::now();

    // AVX2 version
    compute_avx2_soa(x.data(), y.data(), d_avx2.data(), N, A, B, C);
    auto t3 = std::chrono::high_resolution_clock::now();

    // Sample output
    std::cout << "Sample distances (scalar): ";
    for (int i = 0; i < 5; ++i)
        std::cout << d_scalar[i] << " ";
    std::cout << "\n";

    std::cout << "Sample distances (AVX2):   ";
    for (int i = 0; i < 5; ++i)
        std::cout << d_avx2[i] << " ";
    std::cout << "\n";

    // Timing
    std::chrono::duration<double> dt_scalar = t2 - t1;
    std::chrono::duration<double> dt_avx2 = t3 - t2;

    std::cout << "Scalar vectorized duration: " << dt_scalar.count() << " s\n";
    std::cout << "Manual AVX2 duration:       " << dt_avx2.count() << " s\n";

    return 0;
}
