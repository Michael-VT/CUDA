rem for execut compiling RUN command with argument
rem example make from cudaInfo.cu cudaInfo.exe
rem cudamake cudaInfo
del a.out
del a.exe
del a.lib
nvcc -arch=sm_30 %1.cu -ccbin "C:\Program Files (X86)\Microsoft Visual Studio 14.0\VC\bin\x86_arm" --include-path "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"
copy a.exe %1.exe
del a.out
del a.exe
del a.lib
FOR /R /D %I IN (*) DO @ECHO %I





