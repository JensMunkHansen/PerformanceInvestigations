# PerformanceInvestigations

## Installation on Windows

Install the following in the Visual Studio Installer. Required workloads

 * C++ CMake Tools for Windows
 * Desktop development with C++

In the *Individual Components* tab, check

 * CMake support 
 * Clang/LLVM Compiler
   - LLVM with Clang (latest)
 * Ninja Build System
 * MSVC Build Tools (Optional, for compatibility)
   - If you want to use clang-cl (Clang in MSVC compatibility mode), install:
      - MSVC v143 (or latest) C++ toolset
      - Windows SDK (latest)

You may need to update the `CMakePresets.json` with the following:
``` json
"cacheVariables": {
    "CMAKE_C_COMPILER": "C:/Program Files/LLVM/bin/clang.exe",
    "CMAKE_CXX_COMPILER": "C:/Program Files/LLVM/bin/clang++.exe"
}
```
If using clang-cl.exe (MSVC-compatible), set:
``` json
"cacheVariables": {
    "CMAKE_C_COMPILER": "C:/Program Files/LLVM/bin/clang-cl.exe",
    "CMAKE_CXX_COMPILER": "C:/Program Files/LLVM/bin/clang-cl.exe"
}
```
