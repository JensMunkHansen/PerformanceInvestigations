#!/bin/bash

# Ensure an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <release|relwithdebinfo|debug>"
    exit 1
fi

# Convert argument to proper case
BUILD_TYPE=$(echo "$1" | tr '[:lower:]' '[:upper:]')
if [ "$BUILD_TYPE" == "RELEASE" ]; then
    BUILD_TYPE="Release"
elif [ "$BUILD_TYPE" == "DEBUG" ]; then
    BUILD_TYPE="Debug"
elif [ "$BUILD_TYPE" == "RELWITHDEBINFO" ]; then
    BUILD_TYPE="RelWithDebInfo"
else
    echo "Invalid argument: $1"
    echo "Usage: $0 <release|relwithdebinfo|debug>"
    exit 1
fi

# Run CMake commands
cmake --preset linux-gcc
cmake --build build/linux-gcc --config "$BUILD_TYPE" -v
