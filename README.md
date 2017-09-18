# CUDA example with bach file for make
---

example for make asyncAPI.exe from asyncAPI.cu:
---

##cm.bat asyncAPI
---

It contains sample sources and already compiled files.
Before executing the batch file, check for path matching 
in the parameters of the command file. In case of errors, 
search for the name of the file not found, its path and 
substitute it in the line for compilation.
---

for example:
---

set cudbin="C:\Program Files (X86)\Microsoft Visual Studio 14.0\VC\bin\x86_arm"
---

set includ="C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"
---

nvcc -arch=sm_30 %1.cu -ccbin %cudbin% --include-path %includ%
---

I wish you success.
---






