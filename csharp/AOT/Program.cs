using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

public class MatrixMultiplicationBenchmark
{
    const int ProcessorCount = 16;
    public static readonly int[] TestSizes = 
    {
        2 * 16 * ProcessorCount, 
        4 * 16 * ProcessorCount, 
        6 * 16 * ProcessorCount
    };

    private float[] A;
    private float[] B;
    private float[] C;
    private int numThreads;
    private int N;

    public MatrixMultiplicationBenchmark(int matrixSize)
    {
        N = matrixSize;
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

    public double MeasureExecutionTime(Action method, int runs = 10)
    {
        double totalMilliseconds = 0;
        for (int i = 0; i < runs; i++)
        {
            Array.Clear(C, 0, C.Length); // Reset result matrix

            Stopwatch stopwatch = Stopwatch.StartNew();
            method();
            stopwatch.Stop();

            totalMilliseconds += stopwatch.Elapsed.TotalMilliseconds;
        }
        return totalMilliseconds / runs; // Return the average time
    }

    public void SerialMultiplication()
    {
        for (int row = 0; row < N; row++)
            for (int col = 0; col < N; col++)
                for (int idx = 0; idx < N; idx++)
                    C[row * N + col] += A[row * N + idx] * B[idx * N + col];
    }

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
    static void Main()
    {
        Console.WriteLine("Running Matrix Multiplication Benchmarks...\n");

        foreach (int size in MatrixMultiplicationBenchmark.TestSizes)
        {
            Console.WriteLine($"Matrix Size: {size} x {size}");

            var benchmark = new MatrixMultiplicationBenchmark(size);

            double serialTime = benchmark.MeasureExecutionTime(benchmark.SerialMultiplication);
            Console.WriteLine($"  Serial Multiplication: {serialTime:F2} ms");

            double parallelTime = benchmark.MeasureExecutionTime(benchmark.ParallelMultiplication);
            Console.WriteLine($"  Parallel Multiplication: {parallelTime:F2} ms\n");
        }
    }
}
