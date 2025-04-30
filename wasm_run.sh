#!/bin/bash

# Output file
output_file="benchmark_wasm_output.txt"

# Redirect stdout and stderr to the output file
exec > "$output_file" 2>&1

node build/wasm/bin/Release/baseline.cjs --benchmark_min_time=5.0s
node build/wasm/bin/Release/baseline_nonpower.cjs --benchmark_min_time=5.0s
node build/wasm/bin/Release/blocked.cjs --benchmark_min_time=3.0s
node build/wasm/bin/Release/blocked_column.cjs --benchmark_min_time=3.0s
node build/wasm/bin/Release/blocked_column_multi_output.cjs --benchmark_min_time=3.0s
node build/wasm/bin/Release/best.cjs --benchmark_min_time=3.0s

# Linear memory is divided into pages of 64 kB each 

# Working thread should fit into 64 kB pages

# consider
# wasmtime run --mapdir /::/path/to/host/filesystem my_wasm.wasm
