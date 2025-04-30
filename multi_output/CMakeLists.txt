set(TARGET_NAME blocked_column_multi_output)

sps_get_processor_count(MAX_CONCURRENCY)

if (EMSCRIPTEN)
  sps_emscripten_module(
    MAIN
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
  # Set compiler flags for optimization and standards
  add_executable(${TARGET_NAME} benchmark.cpp)
  # Set compiler flags for optimization and standards
  target_link_libraries(${TARGET_NAME} PUBLIC build)
  sps_link_optimization(${TARGET_NAME})
  
  # Link against benchmark and pthread
  target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark Threads::Threads)
endif()

if (NOT EMSCRIPTEN)
  set(TARGET_NAME blocked_column_multi_output_grind)
  add_executable(${TARGET_NAME} grind.cpp)
  target_link_libraries(${TARGET_NAME} PUBLIC build)
  target_include_directories(${TARGET_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/linux")
#  set_target_properties(${TARGET_NAME} PROPERTIES
#    RUNTIME_OUTPUT_NAME grind)

  # cmake .. -DCMAKE_BUILD_TYPE=Release
  # make -f baseline_nonpower/CMakeFiles/baseline_nonpower_grind.dir/build.make baseline_nonpower/CMakeFiles/baseline_nonpower_grind.dir/grind.cpp.s
  # find . -name \*.s | xargs -i -t less {}
    if ("${C_COMPILER_ID}" STREQUAL "Clang")
      add_custom_command(
        TARGET ${TARGET_NAME} POST_BUILD
        COMMAND llvm-objdump -d $<TARGET_FILE:${TARGET_NAME}> > $<TARGET_FILE_DIR:${TARGET_NAME}>/grind.S
        COMMENT "Generating assembly output (.S)"
      )
    elseif(NOT DEFINED MSVC)
      add_custom_command(
        TARGET ${TARGET_NAME} POST_BUILD
        COMMAND objdump -d $<TARGET_FILE:${TARGET_NAME}> > $<TARGET_FILE_DIR:${TARGET_NAME}>/grind.S
        COMMENT "Generating assembly output (.S)"
      )
    endif()
endif()



if (TARGET_NAMES_EQUAL)
  set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_NAME mmul_bench
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
