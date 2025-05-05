using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Perfolizer.Horology;

public class MatrixMultiplicationBenchmark
{
    const int ProcessorCount = 16;

    [Params(2 * 16 * ProcessorCount, 4 * 16 * ProcessorCount, 6 * 16 * ProcessorCount)]
    public int N;

    public float[] A;
    public float[] B;
    public float[] C;
    public int numThreads;

    [GlobalSetup]
    public void Setup()
    {
        numThreads = Environment.ProcessorCount;
        A = new float[N * N];
        B = new float[N * N];
        C = new float[N * N];

        Random rand = new Random(42);
        for (int i = 0; i < N * N; i++)
        {
            A[i] = (float)(rand.NextDouble() * 20 - 10);
            B[i] = (float)(rand.NextDouble() * 20 - 10);
            C[i] = 0;
        }
    }

    [Benchmark]
    public void BlockedParallel()
    {
        MatrixMultAVX2Strict.Multiply(A, B, C, N, numThreads);
    }
}

public static class MatrixMultAVX2Strict
{
    const int TileSize = 128;
    const int BlockSize = 16;
    const int VectorWidth = 8;

    public static void MultiplyScalar(float[] A, float[] B, float[] C, int N)
    {
        Array.Clear(C, 0, N * N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe Vector256<float> Load256(float[] data, int index)
    {
        fixed (float* ptr = &data[index])
            return Avx.LoadVector256(ptr);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void Store256(float[] data, int index, Vector256<float> vec)
    {
        fixed (float* ptr = &data[index])
            Avx.Store(ptr, vec);
    }

    public static void Multiply(float[] A, float[] B, float[] C, int N, int numThreads)
    {
        if (!Avx2.IsSupported)
            throw new PlatformNotSupportedException("AVX2 is required.");

        int tileSize = 128;
        int blockSize = 16;
        int vectorWidth = 8;
        int numTiles = (N + tileSize - 1) / tileSize;
        
        var fullTiles = new List<(int row, int col)>();
        var edgeTiles = new List<(int row, int col)>();
        
        for (int row = 0; row < numTiles; ++row)
        {
            for (int col = 0; col < numTiles; ++col)
            {
                bool isEdge = (row + 1) * tileSize > N || (col + 1) * tileSize > N;
                if (isEdge)
                    edgeTiles.Add((row, col));
                else
                    fullTiles.Add((row, col));
            }
        }
        
        var threadTiles = new List<(int row, int col)>[numThreads];
        for (int i = 0; i < numThreads; ++i)
            threadTiles[i] = new List<(int, int)>();
        
        for (int i = 0; i < fullTiles.Count; ++i)
            threadTiles[i % numThreads].Add(fullTiles[i]);
        
        int tilesPerThread = (edgeTiles.Count + numThreads - 1) / numThreads;
        
        for (int iter = 0 ; iter < 1000 ; iter++)
        {        
            Array.Clear(C, 0, N * N);
        
            var tasks = new Task[numThreads * 2];
        
            // Full (hot) tiles
            for (int t = 0; t < numThreads; ++t)
            {
                int threadId = t;
                tasks[t] = Task.Run(() =>
                {
                    foreach (var (tileRow, tileCol) in threadTiles[threadId])
                    {
                        int rowStart = tileRow * tileSize;
                        int colStart = tileCol * tileSize;
        
                        for (int k = 0; k < N; k += tileSize)
                            ExecuteTileFastAVX2(A, B, C, N, rowStart, colStart, k, tileSize);
                    }
                });
            }
        
            // Edge tiles
            for (int t = 0; t < numThreads; ++t)
            {
                int threadId = t;
                tasks[numThreads + t] = Task.Run(() =>
                {
                    int begin = threadId * tilesPerThread;
                    int end = Math.Min(begin + tilesPerThread, edgeTiles.Count);
                    for (int i = begin; i < end; ++i)
                    {
                        var (tileRow, tileCol) = edgeTiles[i];
                        int rowStart = tileRow * tileSize;
                        int colStart = tileCol * tileSize;
        
                        for (int k = 0; k < N; k += tileSize)
                            ExecuteTileEdge(A, B, C, N, rowStart, colStart, k, tileSize);
                    }
                });
            }
        
            Task.WaitAll(tasks);
            var anchor = System.Threading.Volatile.Read(ref C[0]);
            GC.KeepAlive(anchor); // Ensures read is not optimized away
        }
        
    }
    static void ExecuteTileEdge(float[] A, float[] B, float[] C, int N,
        int rowStart, int colStart, int kStart, int tileSize)
    {
        int rowEnd = Math.Min(rowStart + tileSize, N);
        int colEnd = Math.Min(colStart + tileSize, N);
        int kEnd = Math.Min(kStart + tileSize, N);
    
        const int blockSize = 16;
    
        for (int i = rowStart; i < rowEnd; i += blockSize)
        {
            for (int j = colStart; j < colEnd; j += blockSize)
            {
                float[,] c = new float[blockSize, blockSize];
    
                for (int kk = kStart; kk < kEnd; ++kk)
                {
                    float[] a = new float[blockSize];
                    for (int bi = 0; bi < blockSize; ++bi)
                        a[bi] = (i + bi < N) ? A[(i + bi) * N + kk] : 0.0f;
    
                    for (int bj = 0; bj < blockSize; ++bj)
                    {
                        float b = (j + bj < N) ? B[kk * N + (j + bj)] : 0.0f;
                        for (int bi = 0; bi < blockSize; ++bi)
                        {
                            if ((i + bi < N) && (j + bj < N))
                                c[bi, bj] += a[bi] * b;
                        }
                    }
                }
    
                for (int bi = 0; bi < blockSize; ++bi)
                {
                    for (int bj = 0; bj < blockSize; ++bj)
                    {
                        if ((i + bi < N) && (j + bj < N))
                            C[(i + bi) * N + (j + bj)] += c[bi, bj];
                    }
                }
            }
        }
    }
    
    static unsafe void ExecuteTileFastAVX2(float[] A, float[] B, float[] C, int N,
        int rowStart, int colStart, int kStart, int tileSize)
    {
        int rowEnd = rowStart + tileSize;
        int colEnd = colStart + tileSize;
        int kEnd = kStart + tileSize;
    
        const int blockSize = 16;
        const int vectorWidth = 8;
    
        for (int i = rowStart; i < rowEnd; i += blockSize)
        {
            for (int j = colStart; j < colEnd; j += vectorWidth)
            {
                Vector256<float>[] c = new Vector256<float>[blockSize];
                for (int bi = 0; bi < blockSize; ++bi)
                    c[bi] = Load256(C, (i + bi) * N + j);
    
                for (int kk = kStart; kk < kEnd; ++kk)
                {
                    Vector256<float> bVec = Load256(B, kk * N + j);
    
                    for (int bi = 0; bi < blockSize; ++bi)
                    {
                        float aVal = A[(i + bi) * N + kk];
                        Vector256<float> aVec = Vector256.Create(aVal);
                        c[bi] = Avx.Add(c[bi], Avx.Multiply(aVec, bVec));
                    }
                }
    
                for (int bi = 0; bi < blockSize; ++bi)
                    Store256(C, (i + bi) * N + j, c[bi]);
            }
        }
    }
}


class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("\u26A1 Running manual test...");

            var bench = new MatrixMultiplicationBenchmark
            {
                N = 1024
            };

            bench.Setup();

            var sw = Stopwatch.StartNew();
            bench.BlockedParallel();
            sw.Stop();
            Console.WriteLine($"Parallel: {sw.Elapsed.TotalSeconds:F4} sec");

            return;
        }

        if (args.Length > 0 && args[0] == "--test")
        {
            Console.WriteLine("🧪 Running AVX2 vs Scalar validation test...");
            int N = 512;
            float[] A = new float[N * N];
            float[] B = new float[N * N];
            float[] C1 = new float[N * N];
            float[] C2 = new float[N * N];
    
            Random rand = new Random(123);
            for (int i = 0; i < N * N; i++)
            {
                A[i] = (float)(rand.NextDouble() * 2 - 1);
                B[i] = (float)(rand.NextDouble() * 2 - 1);
            }
    
            MatrixMultAVX2Strict.Multiply(A, B, C1, N, Environment.ProcessorCount);
            MatrixMultAVX2Strict.MultiplyScalar(A, B, C2, N);
    
            int mismatches = 0;
            for (int i = 0; i < N * N; i++)
            {
                if (Math.Abs(C1[i] - C2[i]) > 1e-3f)
                {
                    mismatches++;
                    if (mismatches < 10)
                        Console.WriteLine($"Mismatch at {i}: AVX2={C1[i]}, Scalar={C2[i]}");
                }
            }
    
            Console.WriteLine(mismatches == 0 ? "✅ Validation passed!" : $"❌ {mismatches} mismatches found.");
            return;
        }        
        var config = ManualConfig.Create(DefaultConfig.Instance)
            .AddJob(Job.Default
                .WithId("Net80-Limited")
                .WithRuntime(CoreRuntime.Core80)
                .WithIterationTime(TimeInterval.FromSeconds(30))
                .WithIterationCount(1)
                .WithWarmupCount(1)
                .WithLaunchCount(1))
            .AddExporter(RPlotExporter.Default);

        BenchmarkRunner.Run<MatrixMultiplicationBenchmark>(config);
    }
}
