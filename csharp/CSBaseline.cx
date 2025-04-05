using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;

[SimpleJob(RuntimeMoniker.Net80)]
[RPlotExporter]
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
        Parallel.For(0, numThreads, threadIdx =>
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
        BenchmarkRunner.Run<MatrixMultiplicationBenchmark>();
    }
}
