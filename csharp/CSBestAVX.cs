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

        Array.Clear(C, 0, N * N);

        int numTiles = N / TileSize;
        var threadTiles = new (int, int)[numThreads][];

        for (int t = 0; t < numThreads; ++t)
        {
            if (t >= numTiles)
            {
                threadTiles[t] = Array.Empty<(int, int)>();
                continue;
            }

            var tiles = new (int, int)[numTiles];
            for (int i = 0; i < numTiles; ++i)
            {
                int row = t;
                int col = (t + i) % numTiles;
                tiles[i] = (row, col);
            }
            threadTiles[t] = tiles;
        }

        Task[] tasks = new Task[numThreads];

        for (int t = 0; t < numThreads; ++t)
        {
            int threadId = t;
            tasks[t] = Task.Run(() =>
            {
                foreach (var (tileRow, tileCol) in threadTiles[threadId])
                {
                    int rowStart = tileRow * TileSize;
                    int colStart = tileCol * TileSize;

                    for (int k = 0; k < N; k += TileSize)
                    {
                        for (int i = rowStart; i < rowStart + TileSize; i += BlockSize)
                        {
                            for (int j = colStart; j < colStart + TileSize; j += VectorWidth)
                            {
                                Vector256<float>[] c = new Vector256<float>[BlockSize];
                                for (int bi = 0; bi < BlockSize; ++bi)
                                    c[bi] = Load256(C, (i + bi) * N + j);

                                for (int kk = k; kk < k + TileSize; ++kk)
                                {
                                    for (int bi = 0; bi < BlockSize; ++bi)
                                    {
                                        float aVal = A[(i + bi) * N + kk];
                                        var aVec = Vector256.Create(aVal);
                                        var bVec = Load256(B, kk * N + j);
                                        var mul = Avx.Multiply(aVec, bVec);
                                        c[bi] = Avx.Add(c[bi], mul);
                                    }
                                }

                                for (int bi = 0; bi < BlockSize; ++bi)
                                    Store256(C, (i + bi) * N + j, c[bi]);
                            }
                        }
                    }
                }
            });
        }

        Task.WaitAll(tasks);
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
