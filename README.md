# CUDA examples with batch file to run
This repo contain sample sources and already compiled files.

Example how to make `asyncAPI.exe` from `asyncAPI.cu`:

```shell
cm.bat asyncAPI
```

Also please make sure to provide correct path to cuda binaries and include dir.
For example:

```shell
set cudbin="C:\Program Files (X86)\Microsoft Visual Studio 14.0\VC\bin\x86_arm"
set includ="C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0\common\inc"

nvcc -arch=sm_30 %1.cu -ccbin %cudbin% --include-path %includ%
```
## I wish you success!
