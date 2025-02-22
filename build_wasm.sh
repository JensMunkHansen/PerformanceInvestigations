#!/bin/bash
cmake --preset wasm
cmake --build build/wasm --config Release -v
