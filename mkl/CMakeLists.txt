cmake_minimum_required(VERSION 3.15)
project(benchmark_mkl LANGUAGES CXX)

set(TARGET_NAME intelmkl)

# Find MKL
find_package(MKL QUIET)

if (NOT MKL_FOUND)
  if (EMSCRIPTEN)
    # We have no precompiled MKL for Emscripten
    return()
  endif()
  # Supply MKL_DIR in CMakePresets.json and uncomment at your convenience
  if(NOT EXISTS "${MKL_DIR}")
    return()
  endif()
endif()

# Stupid Intel, we populate it ourselves
set(MKL_INCLUDE_DIR "${MKL_DIR}/../../../include")
set(MKL_LIBRARY_DIR "${MKL_DIR}/../..")
set(MKL_LIBRARIES mkl_rt)
# Define the executable
add_executable(${TARGET_NAME} mkl_sgemm.cpp)

# Include directories for MKL
target_include_directories(${TARGET_NAME} PRIVATE ${MKL_DIR}/../../../include)

# Link MKL and other libraries
target_link_libraries(${TARGET_NAME} PRIVATE ${MKL_LIBRARIES} benchmark::benchmark Threads::Threads)

# Location of Intel libraries
target_link_directories(${TARGET_NAME} PRIVATE ${MKL_LIBRARY_DIR})

if (LINUX)
  # Add runtime library paths for MKL
  set_target_properties(${TARGET_NAME} PROPERTIES
    BUILD_RPATH "${MKL_LIBRARY_DIR}"
    INSTALL_RPATH "${MKL_LIBRARY_DIR}")
else()
  # On windows, you must ensure the MKL dlls's is in your path
endif()

if (TARGET_NAMES_EQUAL)
  set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_NAME mmul_bench
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
