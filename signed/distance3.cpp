#include <cassert>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

// #define USE_MANUAL 0

// Auto-vectorized scalar version (SoA + restrict + ivdep)
void compute_scalar_soa_restrict(const float* __restrict x, const float* __restrict y,
  float* __restrict distances, size_t N, float A, float B, float C)
{
    const float* __restrict x_aligned =
      static_cast<const float* __restrict>(__builtin_assume_aligned(x, 32));
    const float* __restrict y_aligned =
      static_cast<const float* __restrict>(__builtin_assume_aligned(y, 32));
    float* __restrict distances_aligned =
      static_cast<float* __restrict>(__builtin_assume_aligned(distances, 32));

    //    float* __restrict d_aligned = (float* __builtin_assume_aligned(distances, 32));

    float denom = std::sqrt(A * A + B * B);

#pragma GCC ivdep
    for (size_t i = 0; i < N; ++i)
    {
        float xi = x_aligned[i];
        float yi = y_aligned[i];

        float dist = (A * xi + B * yi + C) / denom;
        float yline = (-A * xi - C) / B;

        float sign = (yi < yline) ? -1.0f : 1.0f;
        distances_aligned[i] = dist * sign;
    }
}

int main()
{
    constexpr size_t N = 1 << 20; // 1M points
    alignas(32) std::vector<float> x(N), y(N), d_scalar(N), d_avx2(N);

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
    auto t3 = std::chrono::high_resolution_clock::now();

    // Sample output
    std::cout << "Sample distances (scalar): ";
    for (int i = 0; i < 5; ++i)
        std::cout << d_scalar[i] << " ";
    std::cout << "\n";
    // Timing
    std::chrono::duration<double> dt_scalar = t2 - t1;
    std::cout << "Scalar vectorized duration: " << dt_scalar.count() << " s\n";

    return 0;
}
