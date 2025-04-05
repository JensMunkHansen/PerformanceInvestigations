#!/bin/bash
dotnet new console -o Benchmark
cp MatrixMultiplicationBenchmark.cs ./Benchmark/Program.cs
cd Benchmark
dotnet add package BenchmarkDotNet
dotnet run --configuration Release
dotnet run
