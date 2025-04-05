#!/bin/bash

# Output file
output_file="benchmark_output.txt"

# Redirect stdout and stderr to the output file
exec > "$output_file" 2>&1

./build/linux-gcc/bin/Release/baseline --benchmark_min_time=5.0s
./build/linux-gcc/bin/Release/baseline_nonpower --benchmark_min_time=5.0s
./build/linux-gcc/bin/Release/blocked --benchmark_min_time=3.0s
./build/linux-gcc/bin/Release/blocked_column --benchmark_min_time=3.0s
./build/linux-gcc/bin/Release/blocked_column_multi_output --benchmark_min_time=3.0s
./build/linux-gcc/bin/Release/mmul_bench --benchmark_min_time=3.0s
