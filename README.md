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
 

