# PerformanceInvestigations

## Installation on Windows

Install the following in the Visual Studio Installer. Required workloads

 * C++ CMake Tools for Windows
 * Desktop development with C++

In the *Individual Components* tab, check

 * CMake support 
 * Clang/LLVM Compiler
   - LLVM with Clang (latest)
 * MSVC Build Tools (Optional, for compatibility)
   - If you want to use clang-cl (Clang in MSVC compatibility mode), install:
      - MSVC v143 (or latest) C++ toolset
      - Windows SDK (latest)

 * Copy `CMakeClangPresets.json` to `CMakePresets.json` for Clang usage.
 * Copy `CMakeMSVCPresets.json` to `CMakePresets.json` for MSVC usage.
 * Open folder (this folder in Visual Studio)
 

## Profiling on Windows

* VerySleepy (https://schellcode.github.io/profiling-and-visualization)
* KCachegrind
* [Intel VTune](./VTuneWindows.md)
 - Just use the GUI
 - Command line
```dos
vtune -collect hotspots -result-dir vtune_data ./your_program.exe
vtune -collect memory-access -result-dir vtune_data ./your_program.exe
```

## Profiling on Linux (native)
* [Intel VTune](./VTuneLinux.md)
``` bash
vtune -collect memory-access -result-dir vtune_data ./your_program
vtune -collect memory-access -result-dir vtune_data ./your_program
```

## Simulation on Windows

* [Open source (cross-platform) valgrind](https://sourceforge.net/projects/qcachegrindwin?utm_source=chatgpt.com)
 - Only possible to simulate L1 + LLC
 - Recent version, 2020, https://sourceforge.net/projects/qcachegrind-windows-2020-build/

## Simulation on Linux (and WSL2)
* Cachegrind
``` bash
goto /bin/Debug/
valgrind --tool=cachegrind ./baseline_nonpower_grind
valgrind --tool=callgrind --dump-instr=yes ./baseline_nonpower_grind
cg_annotate on output
```
* Cachegrind (GUI)
 - KCachegrind
 
* Callgrind
``` bash
valgrind --tool=cachegrind \
  --cache-sim=yes \
  --cache=I1:<size>,<associativity>,<line_size> \
  --cache=D1:<size>,<associativity>,<line_size> \
  --cache=LL:<size>,<associativity>,<line_size> \
```
e.g.
``` bash
valgrind --tool=cachegrind --cache-sim=yes   --I1=32768,8,64   --D1=32768,8,64   --LL=1048576,16,64
```

















Percent of samples (we were at this instruction)
Samples in this instruction

You tell the CPU to interrupt after a certain number of events, say every N L3 cache misses
Each sample marks the point where that threshold was reached
The "period" value on a sample is the number of events that triggered that sample

      # TODO: Profile-Guided Optimization (PGO) (10-30%)
      #       Binary Optimization Layout Tool (BOLT) on the final binary (5-20% faster)
      #       bolt binary optimized_binary

