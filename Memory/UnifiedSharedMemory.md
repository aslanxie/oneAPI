# Unified Shared Memory
[Chapter 6 Unified Shared Memory](https://link.springer.com/chapter/10.1007%2F978-1-4842-5574-2_6#DOI) in the guide.  USM defines three different types of allocations: device, host and shared. Here are a simple example on different type.

build command line
```
dpcpp -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device dg2" -o simple-usm-vector simple-usm-vector.cpp
```
running output
```
Runing with vector size: 1048576
shared memory test...
shared memory time consuming：187ms
host memory test...
host memory time consuming：68ms
device memory test...
device memory time consuming：70ms
```

Trace 
```
export SYCL_PI_TRACE=1
```
