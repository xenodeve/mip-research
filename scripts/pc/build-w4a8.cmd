@echo off
rem llama.cpp at 1deefcca3 (C:\AI\llama.cpp) built for sm_89 only: NVFP4 then multiplies with q8_1 activations (W4A8) on both the
rem 4070 Super (Ada) and the 5060 Ti (Blackwell, via PTX JIT), so the two cards run the same kernels. The sm_120 build uses FP4
rem activations (W4A4) on the 5060 Ti.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d C:\AI\llama.cpp
cmake -S . -B build-w4a8 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_BUILD_UI=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" || exit /b 1
cmake --build build-w4a8 --target llama-server llama-perplexity -j 12 || exit /b 2
echo BUILD OK
