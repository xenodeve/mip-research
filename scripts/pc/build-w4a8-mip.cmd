@echo off
rem llama.cpp at 1deefcca3 + the MIP token channel (branch mip-channel), built for sm_89 only exactly as build-w4a8.cmd: build-w4a8-mip.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\AI\llama.cpp
cmake -S . -B build-w4a8-mip -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_BUILD_UI=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" || exit /b 1
cmake --build build-w4a8-mip --target llama-server -j 4 || exit /b 2
echo BUILD OK
