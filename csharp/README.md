AVX-512, .NET8

| Method                 | N    | Mean        | Error     | StdDev     |
|----------------------- |----- |------------:|----------:|-----------:|
| SerialMultiplication   | 512  |   265.46 ms |  0.898 ms |   0.840 ms |
| ParallelMultiplication | 512  |    33.88 ms |  0.505 ms |   0.447 ms |
| SerialMultiplication   | 1024 | 2,353.17 ms |  9.637 ms |   9.014 ms |
| ParallelMultiplication | 1024 |   280.50 ms |  2.561 ms |   2.395 ms |
| SerialMultiplication   | 1536 | 7,942.21 ms | 24.187 ms |  22.624 ms |
| ParallelMultiplication | 1536 | 1,488.12 ms | 34.283 ms | 101.084 ms |

AOT (average over 10 runs)
512   274    ms
512    32.34
1024 2337    ms
1024  266    ms
1536 7812    ms
1536  925    ms

C++ (before we optimize further)
512   114ms
512    12.3
1024 1921ms
1024  172ms
1536 6478ms
1536  514ms

Name;C++;C#;AOT
Serial 512;114ms;265ms;274ms
Serial 1024;1921ms;2353ms;2337ms
Serial 1536;6478ms;7942ms;7812ms

Par 512;12.3ms;33.88ms;32.34ms
Par 1024;172ms;280ms;266ms
Par 1536;514ms;1488ms;925ms
