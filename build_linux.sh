#!/bin/bash
cmake --preset linux
cmake --build build/linux --config Release -v
cmake --build build/linux --config Debug -v
