#include <cassert>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <vector>

struct Point2D
{
    float x, y;
};

// Generate random points
std::vector<Point2D> generate_points(size_t N)
{
    std::vector<Point2D> pts(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (size_t i = 0; i < N; ++i)
    {
        pts[i].x = dist(rng);
        pts[i].y = dist(rng);
    }
    return pts;
}

// Scalar version — lets the compiler auto-vectorize
void compute_scalar(
  const std::vector<Point2D>& points, std::vector<float>& distances, float A, float B, float C)
{
    float denom = std::sqrt(A * A + B * B);
    for (size_t i = 0; i < points.size(); ++i)
    {
        float x = points[i].x;
        float y = points[i].y;
        float dist = (A * x + B * y + C) / denom;
        float yline = (-A * x - C) / B;
        distances[i] = (y < yline) ? -dist : dist;
    }
}

// Manual AVX2 version
void compute_avx2(
  const std::vector<Point2D>& points, std::vector<float>& distances, float A, float B, float C)
{
    assert(points.size() == distances.size());
    size_t N = points.size();
    size_t i = 0;

    const __m256 a_vec = _mm256_set1_ps(A);
    const __m256 b_vec = _mm256_set1_ps(B);
    const __m256 c_vec = _mm256_set1_ps(C);
    const __m256 denom = _mm256_set1_ps(std::sqrt(A * A + B * B));

    for (; i + 7 < N; i += 8)
    {
        __m256 x = _mm256_set_ps(points[i + 7].x, points[i + 6].x, points[i + 5].x, points[i + 4].x,
          points[i + 3].x, points[i + 2].x, points[i + 1].x, points[i + 0].x);

        __m256 y = _mm256_set_ps(points[i + 7].y, points[i + 6].y, points[i + 5].y, points[i + 4].y,
          points[i + 3].y, points[i + 2].y, points[i + 1].y, points[i + 0].y);

        __m256 ax = _mm256_mul_ps(a_vec, x);
        __m256 by = _mm256_mul_ps(b_vec, y);
        __m256 numerator = _mm256_add_ps(_mm256_add_ps(ax, by), c_vec);
        __m256 dist = _mm256_div_ps(numerator, denom);

        __m256 neg_ax = _mm256_sub_ps(_mm256_setzero_ps(), ax); // -Ax
        __m256 rhs = _mm256_div_ps(_mm256_sub_ps(neg_ax, c_vec), b_vec);
        __m256 mask = _mm256_cmp_ps(y, rhs, _CMP_LT_OS);

        __m256 neg_dist = _mm256_sub_ps(_mm256_setzero_ps(), dist);
        __m256 result = _mm256_blendv_ps(dist, neg_dist, mask);

        _mm256_storeu_ps(&distances[i], result);
    }

    // Scalar tail
    for (; i < N; ++i)
    {
        float x = points[i].x, y = points[i].y;
        float dist = (A * x + B * y + C) / std::sqrt(A * A + B * B);
        float yline = (-A * x - C) / B;
        distances[i] = (y < yline) ? -dist : dist;
    }
}

int main()
{
    constexpr size_t N = 1 << 20; // 1M points
    auto points = generate_points(N);
    std::vector<float> dist_scalar(N), dist_avx2(N);

    float A = -0.5f, B = 1.0f, C = -10.0f;

    auto t1 = std::chrono::high_resolution_clock::now();
    compute_scalar(points, dist_scalar, A, B, C);
    auto t2 = std::chrono::high_resolution_clock::now();
    compute_avx2(points, dist_avx2, A, B, C);
    auto t3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> dt_scalar = t2 - t1;
    std::chrono::duration<double> dt_avx2 = t3 - t2;

    std::cout << "Sample distances (scalar): ";
    for (int i = 0; i < 5; ++i)
        std::cout << dist_scalar[i] << " ";
    std::cout << "\n";

    std::cout << "Sample distances (AVX2):   ";
    for (int i = 0; i < 5; ++i)
        std::cout << dist_avx2[i] << " ";
    std::cout << "\n";

    std::cout << "Scalar duration: " << dt_scalar.count() << " s\n";
    std::cout << "AVX2 duration:   " << dt_avx2.count() << " s\n";

    return 0;
}
