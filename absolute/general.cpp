inline void execute_tile_fast(const float* __restrict A, const float* __restrict B,
  float* __restrict C, std::size_t L, std::size_t M, std::size_t N, std::size_t row_start,
  std::size_t col_start, std::size_t k, std::size_t tile_size)
{
    const std::size_t row_end = std::min(row_start + tile_size, L);
    const std::size_t col_end = std::min(col_start + tile_size, N);
    const std::size_t k_end = std::min(k + tile_size, M);
    const std::size_t block_size = 16;

    for (std::size_t i = row_start; i < row_end; i += block_size)
    {
        for (std::size_t j = col_start; j < col_end; j += block_size)
        {
            alignas(64) float c[block_size][block_size] = { 0 };

            for (std::size_t kk = k; kk < k_end; ++kk)
            {
                alignas(64) float a[block_size];
#pragma ivdep
                for (std::size_t bi = 0; bi < block_size; ++bi)
                    a[bi] = (i + bi < L) ? A[(i + bi) * M + kk] : 0.0f;

#pragma ivdep
                for (std::size_t bj = 0; bj < block_size; ++bj)
                {
                    float b = (j + bj < N) ? B[kk * N + (j + bj)] : 0.0f;
#pragma ivdep
                    for (std::size_t bi = 0; bi < block_size; ++bi)
                        c[bi][bj] += a[bi] * b;
                }
            }

#pragma ivdep
            for (std::size_t bi = 0; bi < block_size; ++bi)
                for (std::size_t bj = 0; bj < block_size; ++bj)
                    if ((i + bi) < L && (j + bj) < N)
                        C[(i + bi) * N + (j + bj)] += c[bi][bj];
        }
    }
}

static void tiled_blocked_parallel_mmul_general(const float* A, const float* B, float* C,
  std::size_t L, std::size_t M, std::size_t N, std::size_t num_threads, ThreadPool& pool)
{
    const std::size_t tile_size = 128;
    const std::size_t num_tile_rows = (L + tile_size - 1) / tile_size;
    const std::size_t num_tile_cols = (N + tile_size - 1) / tile_size;

    // Classify tiles: full vs edge
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

    // Diagonal round-robin assignment for full tiles
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> thread_tiles(num_threads);
    for (std::size_t i = 0; i < full_tiles.size(); ++i)
    {
        std::size_t t = (full_tiles[i].first + full_tiles[i].second) % num_threads;
        thread_tiles[t].push_back(full_tiles[i]);
    }

    std::vector<std::future<void>> futures;

    // Launch full (hot) tiles first
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
                      execute_tile_fast(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          }));
    }

    // Launch edge (cold) tiles after all hot tiles
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
                  const auto& [tile_row, tile_col] = edge_tiles[i];
                  std::size_t row_start = tile_row * tile_size;
                  std::size_t col_start = tile_col * tile_size;

                  for (std::size_t k = 0; k < M; k += tile_size)
                  {
                      execute_tile_fast(A, B, C, L, M, N, row_start, col_start, k, tile_size);
                  }
              }
          }));
    }

    for (auto& fut : futures)
        fut.get();
}

static void tiled_blocked_parallel_mmul_bench(benchmark::State& s)
{
    const std::size_t L = s.range(0);
    const std::size_t M = s.range(1);
    const std::size_t N = s.range(2);

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
        tiled_blocked_parallel_mmul_general(A, B, C, L, M, N, numThreads, pool);
    }

    ALIGNED_FREE(A);
    ALIGNED_FREE(B);
    ALIGNED_FREE(C);
}
