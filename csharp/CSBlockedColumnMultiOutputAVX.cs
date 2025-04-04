using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices; // Required for MemoryMarshal
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
    public void BlockedSerial()
    {
        BlockedColumnMultiOutputMmul(A, B, C, N);
    }

    [Benchmark]
    public void BlockedParallel()
    {
        BlockedColumnMultiOutputParallelMmulFull(A, B, C, N);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe Vector256<float> LoadVector256(float[] data, int index)
    {
        fixed (float* ptr = &data[index])
        {
            return Avx.LoadVector256(ptr);
        }
    }
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static unsafe void StoreVector256(float[] data, int index, Vector256<float> value)
    {
        fixed (float* ptr = &data[index])
        {
            Avx.Store(ptr, value);
        }
    }
    
    public static void BlockedColumnMultiOutputMmul(float[] A, float[] B, float[] C, int N)
    {
        const int BLOCK = 16;

        for (int colChunk = 0; colChunk < N; colChunk += BLOCK)
        {
            for (int rowChunk = 0; rowChunk < N; rowChunk += BLOCK)
            {
                for (int tile = 0; tile < N; tile += BLOCK)
                {
                    for (int row = 0; row < BLOCK; row++)
                    {
                        int rowIndex = row + rowChunk;
                        if (rowIndex >= N) continue;

                        for (int tileRow = 0; tileRow < BLOCK; tileRow++)
                        {
                            int tileIndex = tile + tileRow;
                            if (tileIndex >= N) continue;

                            float aVal = A[rowIndex * N + tileIndex];

                            if (Avx2.IsSupported)
                            {
                                for (int idx = 0; idx < BLOCK; idx += 8)
                                {
                                    int colIndex = colChunk + idx;
                                    if (colIndex + 7 >= N) continue;

                                    int bIndex = tileIndex * N + colIndex;
                                    int cIndex = rowIndex * N + colIndex;

                                    var bVec = LoadVector256(B, bIndex);
                                    var cVec = LoadVector256(C, cIndex);
                                    var result = Avx.Multiply(Vector256.Create(aVal), bVec);
                                    result = Avx.Add(result, cVec);
                                    StoreVector256(C, cIndex, result);
                                }
                            }
                            else
                            {
                                for (int idx = 0; idx < BLOCK; idx++)
                                {
                                    int colIndex = colChunk + idx;
                                    if (colIndex >= N) continue;

                                    int bIndex = tileIndex * N + colIndex;
                                    int cIndex = rowIndex * N + colIndex;

                                    C[cIndex] += aVal * B[bIndex];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void BlockedColumnMultiOutputParallelMmul(float[] A, float[] B, float[] C, int N, int startCol, int endCol)
    {
        const int BLOCK = 16;

        for (int colChunk = startCol; colChunk < endCol; colChunk += BLOCK)
        {
            for (int rowChunk = 0; rowChunk < N; rowChunk += BLOCK)
            {
                for (int tile = 0; tile < N; tile += BLOCK)
                {
                    for (int row = 0; row < BLOCK; row++)
                    {
                        int rowIndex = row + rowChunk;
                        if (rowIndex >= N) continue;

                        for (int tileRow = 0; tileRow < BLOCK; tileRow++)
                        {
                            int tileIndex = tile + tileRow;
                            if (tileIndex >= N) continue;

                            float aVal = A[rowIndex * N + tileIndex];

                            if (Avx2.IsSupported)
                            {
                                for (int idx = 0; idx < BLOCK; idx += 8)
                                {
                                    int colIndex = colChunk + idx;
                                    if (colIndex + 7 >= N) continue;

                                    int bIndex = tileIndex * N + colIndex;
                                    int cIndex = rowIndex * N + colIndex;

                                    var bVec = LoadVector256(B, bIndex);
                                    var cVec = LoadVector256(C, cIndex);
                                    var result = Avx.Multiply(Vector256.Create(aVal), bVec);
                                    result = Avx.Add(result, cVec);
                                    StoreVector256(C, cIndex, result);
                                }
                            }
                            else
                            {
                                for (int idx = 0; idx < BLOCK; idx++)
                                {
                                    int colIndex = colChunk + idx;
                                    if (colIndex >= N) continue;

                                    int bIndex = tileIndex * N + colIndex;
                                    int cIndex = rowIndex * N + colIndex;

                                    C[cIndex] += aVal * B[bIndex];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static void BlockedColumnMultiOutputParallelMmulFull(float[] A, float[] B, float[] C, int N)
    {
        int numChunks = N / 16 + (N % 16 == 0 ? 0 : 1);
        Parallel.For(0, numChunks, chunkIdx =>
        {
            int startCol = chunkIdx * 16;
            int endCol = Math.Min(N, startCol + 16);
            BlockedColumnMultiOutputParallelMmul(A, B, C, N, startCol, endCol);
        });
    }
}

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("⚡ Running manual test...");

            var bench = new MatrixMultiplicationBenchmark
            {
                N = 1024
            };

            bench.Setup();

            var sw = Stopwatch.StartNew();
            bench.BlockedSerial();
            sw.Stop();
            Console.WriteLine($"Serial: {sw.Elapsed.TotalSeconds:F4} sec");

            bench.Setup();
            sw.Restart();
            bench.BlockedParallel();
            sw.Stop();
            Console.WriteLine($"Parallel: {sw.Elapsed.TotalSeconds:F4} sec");

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
