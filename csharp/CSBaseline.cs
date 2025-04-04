using System;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Environments;
using Perfolizer.Horology; // ✅ Needed for TimeInterval

public class MatrixMultiplicationBenchmark
{
    const int ProcessorCount = 16;

    [Params(2 * 16 * ProcessorCount, 4 * 16 * ProcessorCount, 6 * 16 * ProcessorCount)]
    public int N;

    private float[] A;
    private float[] B;
    private float[] C;
    private int numThreads;

    [GlobalSetup]
    public void Setup()
    {
        numThreads = Environment.ProcessorCount;
        A = new float[N * N];
        B = new float[N * N];
        C = new float[N * N];

        Random rand = new Random();
        for (int i = 0; i < N * N; i++)
        {
            A[i] = (float)(rand.NextDouble() * 20 - 10);
            B[i] = (float)(rand.NextDouble() * 20 - 10);
            C[i] = 0;
        }
    }

    [Benchmark]
    public void SerialMultiplication()
    {
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                for (int idx = 0; idx < N; idx++)
                    C[row * N + col] += A[row * N + idx] * B[idx * N + col];
    }

    [Benchmark]
    public void ParallelMultiplication()
    {
        int chunkSize = N / numThreads;
        System.Threading.Tasks.Parallel.For(0, numThreads, threadIdx =>
        {
            int startRow = threadIdx * chunkSize;
            int endRow = (threadIdx == numThreads - 1) ? N : startRow + chunkSize;

            for (int row = startRow; row < endRow; row++)
                for (int col = 0; col < N; col++)
                    for (int idx = 0; idx < N; idx++)
                        C[row * N + col] += A[row * N + idx] * B[idx * N + col];
        });
    }
}

class Program
{
static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("⚡ Running raw manual benchmark (no BenchmarkDotNet)");

            var bench = new MatrixMultiplicationBenchmark
            {
                N = 1024 // Set a fixed size for manual test
            };

            bench.Setup();

            var sw = Stopwatch.StartNew();
            bench.SerialMultiplication();
            sw.Stop();
            Console.WriteLine($"Serial: {sw.Elapsed.TotalSeconds:F4} sec");

            bench.Setup(); // Re-init data
            sw.Restart();
            bench.ParallelMultiplication();
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
