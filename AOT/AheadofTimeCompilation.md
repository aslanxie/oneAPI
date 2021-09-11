# Ahead of Time Compilation and Memory Type

## Ahead of Time Compilation
It moves device code compilation from running time, no additional compilation time is done when running your application. How much time can be reduced depends on the complexity of device codes.

Building without AOT compilation 
```
dpcpp -fsycl -o simple-vector simple-vector.cpp
```
Building with AOT compilation 
```
dpcpp -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device *" -o simple-vector simple-vector.cpp
```

## Shared Memory and Device Memory
In SYCL applicaiton codes level, with Unified Shared Memory, applicaiton can operate the same buffer and need not care it's on host or device. 
It's very conveniently for coding, but the cost is obviously, too.
In this test with 1M data, time on memory from ```malloc_device```is 70ms,  and ```malloc_shared``` is 193ms.

# Reference
1.[Ahead of Time Compilation](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html)

2.[shared memory and device memory](https://github.com/jeffhammond/dpcpp-tutorial)

3. [oneAPI DPC++ Compiler and Runtime architecture design](https://intel.github.io/llvm-docs/CompilerAndRuntimeDesign.html)
