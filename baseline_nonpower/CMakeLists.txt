set(TARGET_NAME baseline_nonpower)

if (EMSCRIPTEN)
  sps_emscripten_module(
    MAIN    
    TARGET_NAME ${TARGET_NAME}
    SOURCE_FILES benchmark.cpp
    THREAD_POOL_SIZE ${MAX_CONCURRENCY}
    ES6_MODULE OFF
    EXIT_RUNTIME ON
    OPTIMIZATION BEST
    COMPILE_OPTIMIZATION BEST
    DEBUG NONE
    LIBRARIES build;benchmark::benchmark;Threads::Threads
    FILE_SYSTEM ON
    PRE_JS ${PROJECT_SOURCE_DIR}/emscripten/pre.js
    THREADING_ENABLED ON)
else()
  add_executable(${TARGET_NAME} benchmark.cpp)
  # Set compiler flags for optimization and standards
  target_link_libraries(${TARGET_NAME} PUBLIC build)
  # Link against benchmark and pthread
  target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark Threads::Threads)
endif()

if (NOT EMSCRIPTEN)
  set(TARGET_NAME baseline_nonpower_grind)
  add_executable(${TARGET_NAME} grind.cpp)
  target_link_libraries(${TARGET_NAME} PUBLIC build)
  set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_NAME grind)

  if (CMAKE_CONFIGURATION_TYPES)
    add_custom_command(
      TARGET ${TARGET_NAME} POST_BUILD
      COMMAND objdump -d $<TARGET_FILE:${TARGET_NAME}> > $<TARGET_FILE_DIR:${TARGET_NAME}>/grind.s
      COMMENT "Generating assembly output (.s)"
    )
  else()
    add_custom_command(
      TARGET ${TARGET_NAME} POST_BUILD
      COMMAND objdump -d $<TARGET_FILE:${TARGET_NAME}> > $<TARGET_FILE_DIR:${TARGET_NAME}>/grind.s
      COMMENT "Generating assembly output (.s) for WebAssembly"
    )
  endif()
endif()
