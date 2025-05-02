#!/bin/bash

# Output file
output_file="benchmark_clang_output.txt"

# Redirect stdout and stderr to the output file
exec > "$output_file" 2>&1

./build/linux/bin/Release/baseline --benchmark_min_time=5.0s
./build/linux/bin/Release/baseline_nonpower --benchmark_min_time=5.0s
./build/linux/bin/Release/blocked --benchmark_min_time=3.0s
./build/linux/bin/Release/blocked_column --benchmark_min_time=3.0s
./build/linux/bin/Release/blocked_column_multi_output --benchmark_min_time=3.0s
./build/linux/bin/Release/blocked_column_multi_output_accumulate --benchmark_min_time=3.0s
./build/linux/bin/Release/intelmkl --benchmark_min_time=3.0s


#for CPUFREQ in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
#  echo performance | sudo tee $CPUFREQ
#done
