set(TARGET_NAME baseline)

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
else()
  add_executable(${TARGET_NAME} benchmark.cpp)
  # Set compiler flags for optimization and standards
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-O3 -march=native -mtune=native -flto -fuse-linker-plugin -ftree-vectorize -fopt-info-vec>
    $<$<CXX_COMPILER_ID:Clang>:-O3 -march=native -Rpass=loop-vectorize -Rpass-missed=loop-vectorize>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /GL /fp:fast /Qvec /Qpar /arch:AVX2>
  )
  target_link_options(${TARGET_NAME} PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-fuse-ld=lld -flto>
  )  
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
  
  # Link against benchmark and pthread
  target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark Threads::Threads)
endif()
