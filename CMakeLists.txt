cmake_minimum_required(VERSION 3.15)

project(PerformanceInvestigations
LANGUAGES C CXX)

option(TARGET_NAMES_EQUAL "Same target names" OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

# Standard install dirs (ADR)
include(GNUInstallDirs)

# Set up our directory structure for output libraries and binaries
# (Note: these are the build locations, not the install locations)
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PerformanceInvestigations_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
endif()

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  if(UNIX)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PerformanceInvestigations_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
  else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PerformanceInvestigations_BINARY_DIR}${CMAKE_INSTALL_BINDIR}")
  endif()
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PerformanceInvestigations_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

include(spsBenchmark)
include(spsEmscriptenSettings)
include(spsHardware)

add_library(build INTERFACE)
include(spsCompilerPlatformFlags)

add_subdirectory(baseline)
add_subdirectory(baseline_nonpower)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  return()
endif()

# Needs to be tested on windows
add_subdirectory(blocked)
add_subdirectory(blocked_column) 
add_subdirectory(blocked_column_multi_output)
add_subdirectory(mkl)
# L1 cache, transpose and multiply
