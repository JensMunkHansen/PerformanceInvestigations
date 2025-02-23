set(TARGET_NAME blocked_column)

sps_get_processor_count(MAX_CONCURRENCY)

if (EMSCRIPTEN)
  sps_emscripten_module(
    TARGET_NAME ${TARGET_NAME}
    SOURCE_FILES benchmark.cpp
    THREAD_POOL_SIZE ${MAX_CONCURRENCY}
    ES6_MODULE OFF
    EXIT_RUNTIME ON
    OPTIMIZATION BEST
    LIBRARIES benchmark::benchmark;Threads::Threads
    FILE_SYSTEM ON
    PRE_JS ${PROJECT_SOURCE_DIR}/emscripten/pre.js
    THREADING_ENABLED ON)

sps_target_compile_flags(${TARGET_NAME}
  THREADING_ENABLED ON
  OPTIMIZATION BEST)
  
else()
  add_executable(${TARGET_NAME} benchmark.cpp)
  # Set compiler flags for optimization and standards
  target_link_libraries(${TARGET_NAME} PUBLIC build)
  # Link against benchmark and pthread
  target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark Threads::Threads)
endif()

if (TARGET_NAMES_EQUAL)
set_target_properties(${TARGET_NAME} PROPERTIES
  RUNTIME_OUTPUT_NAME mmul_bench
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
