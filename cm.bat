@ECHO OFF
rem cls
color 0A
color 03
IF NOT EXIST %1.cu ECHO File %1.cu not exist
IF NOT EXIST %1.cu EXIT /B 0
rem for execut compiling RUN command with argument
rem example make from cudaInfo.cu cudaInfo.exe
rem cudamake cudaInfo
rem nvcc -arch=sm_30 %1.cu -ccbin "C:\Program Files (X86)\Microsoft Visual Studio 14.0\VC\bin\x86_arm" --include-path "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"
rem Задать путь к линковщику Visual Studio
set cudbin="C:\Program Files (X86)\Microsoft Visual Studio 14.0\VC\bin\x86_arm"
rem Задать путь к файлам заголовков NVIDIA CUDA
set includ="C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"
@ECHO ON
nvcc -arch=sm_30 %1.cu -ccbin %cudbin% --include-path %includ%
@ECHO OFF
copy a.exe %1.exe
IF EXIST a.out del a.out
IF EXIST a.exe del a.exe
IF EXIST a.exp del a.exp
IF EXIST a.lib del a.lib
IF EXIST %1.log del %1.log 
IF EXIST %1.exe %1.exe > %1.log 







