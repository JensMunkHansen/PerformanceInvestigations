#!/bin/bash
cmake --preset wasm -DPerformanceInvestigations_WASM_SIMD=ON
cmake --build build/wasm --config Release -v 
