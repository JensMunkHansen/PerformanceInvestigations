#!/bin/bash
cmake --preset linux-gcc
cmake --build build/linux-gcc --config Release -v
cmake --build build/linux-gcc --config Debug -v
