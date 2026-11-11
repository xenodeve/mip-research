@echo off
rem Build the CUDAMathBackend DLL for both PC cards: sm_89 (RTX 4070 SUPER, Ada) and sm_120 (RTX 5060 Ti, Blackwell), each as real
rem machine code, plus PTX for later cards. Output: build-cuda\mipmath.dll beside this repository's root (ignored by git).
rem The corpus must pass on both cards (design "Both cards"); integer code only, so the two cards run the same arithmetic.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set HERE=%~dp0
set OUT=%HERE%..\..\build-cuda
if not exist "%OUT%" mkdir "%OUT%"
%NVCC% -O3 -std=c++17 -shared -o "%OUT%\mipmath.dll" "%HERE%mipmath.cu" ^
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 ^
  -Xcompiler "/MD /W3" -lineinfo || exit /b 1
echo BUILD OK %OUT%\mipmath.dll
