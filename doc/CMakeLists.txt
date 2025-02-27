cmake_minimum_required(VERSION 3.14)

project(mmul
  LANGUAGES
  CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../CMake)

find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)

include(spsEmscriptenSettings)
include(spsHardware)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG v1.8.2
    GIT_SHALLOW ON
    GIT_PROGRESS ON
    FIND_PACKAGE_ARGS 1.8.2)

add_subdirectory(baseline)
add_subdirectory(baseline_nonpower)
add_subdirectory(blocked)
add_subdirectory(blocked_column)
add_subdirectory(blocked_column_multi_output)
if (NOT EMSCRIPTEN)
  add_subdirectory(mkl)
endif()
