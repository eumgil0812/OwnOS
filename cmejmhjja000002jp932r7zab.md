---
title: "Cuda Start!"
datePublished: Wed Aug 20 2025 06:59:50 GMT+0000 (Coordinated Universal Time)
cuid: cmejmhjja000002jp932r7zab
slug: cuda-start
tags: cuda, hpc

---

# Installing the CUDA Toolkit and Driver

To use CUDA, you basically need a laptop or desktop equipped with an NVIDIA GPU. For this purpose, I purchased a new Linux-based laptop with the required graphics hardware.

* **CUDA Toolkit** provides the compiler (`nvcc`), libraries, and sample codes required to build and run CUDA applications.
    
* **NVIDIA Driver** is necessary for the operating system to communicate properly with the GPU hardware.
    

The CUDA Toolkit can be downloaded directly from NVIDIA’s official website.

[https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)

The important thing is to get the installation right on the first try. When I reinstalled CUDA because I wasn’t satisfied with some parts of the initial setup, things quickly became messy.

If you ever need to reinstall, you have to completely remove all related files and data from the system. Tracking down and deleting everything scattered across different directories wasn’t an easy task.

That’s why I strongly recommend making sure the first installation is done properly, so you can avoid the hassle of reinstalling later.

# Downloading the CUDA Samples

After installing the CUDA Toolkit, you can also use the official CUDA sample codes provided by NVIDIA. These examples are very useful for testing your setup and learning how to write CUDA programs.

If the samples are not included with your installation, you can download them directly from NVIDIA’s GitHub repository:

```c
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
```

Let’s try running the example located in `cuda-samples/Samples/1_Utilities/deviceQuery`.

The `deviceQuery` example is one of the test programs provided with the CUDA Toolkit.  
Its purpose is to check whether your GPU is properly recognized by CUDA and to display its detailed specifications.

```bash
# Current location: cuda-samples/Samples/1_Utilities/deviceQuery
mkdir -p build && cd build

# It’s cleaner to explicitly specify the CUDA compiler path
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# Compile
cmake --build . -j

# Run
./deviceQuery
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755667001955/a62e0fd7-a0d3-4f20-bf06-0f66a3a7b94a.png align="center")

# Basic Knowledge about CUDA

## 1\. Host and Device

In documents related to CUDA and GPU programming, the terms **host** and **device** frequently appear.

* The **host** generally refers to the CPU.
    
* The **device** refers to the GPU.
    
* Since the first GPU module (kernel) is launched by the CPU, the CPU is considered the host.
    

For example, host code means code that runs on the CPU, while device code refers to code that runs on the GPU.

---

### 2\. CUDA Program

A CUDA program consists of both **host code** and **device code**. Since the first device code (kernel) executed on the GPU must be launched from the host code, a CUDA program always requires host code.

The traditional file extensions used for writing CUDA programs are **.cu** and **.cuh**.

# Naturally, the greatest beginning is 'Hello'.

## 1\. Hello CUDA

```cpp
  1 #include "cuda_runtime.h"
  2 #include "device_launch_parameters.h"
  3 #include <stdio.h>
  4 
  5 __global__ void helloCUDA(void)
  6 {
  7     printf("Hello CUDA from GPU!\n");
  8 }
  9 
 10 int main(void)
 11 {
 12     printf("Hello GPU from CPU!\n");
 13     helloCUDA<<<1, 10>>>();
 14     cudaDeviceSynchronize(); // GPU 실행이 끝날 때까지 대기
 15     return 0;
 16 }
 17
```

```bash
nvcc hello.cu -o hello
./hello
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755670035759/9b941a08-7e66-4364-8dfc-887892f30afe.png align="center")

The first two header files are part of the CUDA Runtime API, which contain important definitions for writing programs.

You can think of them as something you should almost always include.

## 2\. CUDA C/C++ Keyword

CUDA is a programming interface that extends C/C++. To enable this extension, CUDA uses C/C++ keywords.

```cpp
  5 __global__ void helloCUDA(void)
  6 {
  7     printf("Hello CUDA from GPU!\n");
  8 }
```

`__global__` specifies a function that is called from the host and executed on the device.

| Keyword | *Function Caller* | *Execution Space* |
| --- | --- | --- |
| `__host__` | host | host |
| `__device__` | device | device |
| `__global__` | host | device |

By default, if no keyword is provided, a function is treated as a `__host__` function.  
When you need a function to run on both the host and the device, you can declare it with both `__host__` and `__device__`.  
To enable the host to call a function that runs on the device, CUDA provides the `__global__` keyword, which is used for launching kernels.

## 3\. CUDA Kernel & Thread Hierarchy

A **CUDA kernel** is a function that defines the behavior of CUDA threads and serves as the channel through which the **host (CPU)** requests computations to be performed on the **device (GPU)**.

When launching a kernel, you must specify the **number of threads** that will execute the computation. This is done using the special **execution configuration syntax** `<<< >>>`.

CUDA threads are not launched individually but are organized into **groups**, and these groups are further arranged in a hierarchical structure:

* **Thread**: the smallest execution unit.
    
* **Block**: a group of threads. Each block can contain up to thousands of threads (depending on hardware limits).
    
* **Grid**: a collection of blocks.
    

Thus, the hierarchy can be summarized as:

**Grid → Blocks → Threads**

```cpp
 13     helloCUDA<<<1, 10>>>();
```