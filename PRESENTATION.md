# Presentation

Compiler:
 - RelWithDebInfo
   * Some optimization
   * Profiling
   * Stack trace
 - Release
   * Inlining
   * Re-ordering
   * Loop un-rolling
   * Vectorization
   * ...
   * Binary Optimization Layout Tool (5-20%)
   * Profile Guided Optimization (10-30%)
   * ML-guided optimization (deep learning?)
  - What to inline, loop-unrolling, register allocation, instruction scheduling
 
 - Debug
   * Baseline

Profiling:
 - Sampling (real)
   * Time, Cache misses
   * Instructions retired (grouped)
   * Limited to what the CPU exposes
   * perf (Linux and WSL2), `perf list`
   * VTune (Intel only)
 - Simulation
   * Cache and branch prediction (`cachegrind`)

Memory:
 - Data access (our topic)
 - Instruction flow (a bit more tricky)

Coffee before architecture:

Matrix multiplication

Show results:
 - C++/C# (3x)
 - Threading 16 -> 8
 - More threads (5e6)
 - 1024/1040

Cache collissions: L1 48kB (8-way)

Hardware

Show perf stat
Show cachegrind

Blocked:

Blocked-column:

Multiple outputs:

sA + sB + sC <= 48 kB
transpose sB on first access
prefetch
