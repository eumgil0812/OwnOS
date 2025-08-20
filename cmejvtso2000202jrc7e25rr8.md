---
title: "Basic CUDA Memory APIs"
datePublished: Wed Aug 20 2025 11:21:18 GMT+0000 (Coordinated Universal Time)
cuid: cmejvtso2000202jrc7e25rr8
slug: basic-cuda-memory-apis
tags: cudaapi

---

In CUDA programming, data transfer between **host memory** and **device memory** is a fundamental step. In this section, we’ll look at the basics of how to allocate device memory using CUDA APIs and how to copy data between the host and the device.

# 1\. Device Memory Allocation and Initialization APIs

To copy data into device memory, you must first allocate space on the device.  
This is conceptually the same as using `malloc()` in C to allocate memory on the host.

## **(a) Device Memory Allocation –** `cudaMalloc()`

In CUDA, before you can use device memory, you need to allocate GPU memory.  
The API used for this purpose is `cudaMalloc()`.

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

* **devPtr**: A pointer to the allocated device memory
    
* **size**: The size of the memory to allocate, in bytes
    
* **Return value**: `cudaSuccess` if successful, or an error code if it fails
    

```cpp
#include <stdio.h>
 #include <cuda_runtime.h>
  
  int main() {
      int *d_array;   // Pointer to device memory
      size_t size = 10 * sizeof(int);
  
      // Allocate GPU memory
      cudaError_t err = cudaMalloc((void**)&d_array, size);
  
      if (err != cudaSuccess) {
          printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
          return -1;
      }
  
      printf("Device memory allocated successfully!\n");
  
      // Free the memory after use
      cudaFree(d_array);
  
      return 0;
  }
```

* It is important to note that the address pointed to by `d_array` refers to device memory, and therefore cannot be directly accessed from the host code.
    
    Tip. In CUDA programs, it is common practice to distinguish variables that use host memory from those that use device memory by prefixing variables in device memory with the letter `d`, which stands for “device.”
    

## **(b) Device Memory Deallocation –** `cudaFree()`

This step involves freeing device memory.  
Any memory allocated with `cudaMalloc()` must be released with `cudaFree()` once it is no longer needed.  
Otherwise, a memory leak in GPU memory may occur, which can cause subsequent kernel executions or other memory allocations to fail.

```cpp
cudaError_t cudaFree(void* devPtr);
```

```cpp
int *d_array;
size_t size = 100 * sizeof(int);

// 1. Allocate device memory
cudaMalloc((void**)&d_array, size);

// ... (use d_array in kernel computations)

// 2. Free device memory
cudaError_t err = cudaFree(d_array);

if (err != cudaSuccess) {
    printf("cudaFree failed: %s\n", cudaGetErrorString(err));
} else {
    printf("Device memory successfully freed!\n");
}
```

---

## (c) Device Memory Initialization – `cudaMemset()`

* The API used to set initial values after allocating device memory is `cudaMemset()`.  
    It works similarly to the C language `memset()` function, filling values byte by byte.
    
    ```cpp
    cudaError_t cudaMemset(void* devPtr, int value, size_t count);
    ```
    
    * **devPtr**: A pointer to the device memory
        
    * **value**: The value to set (in bytes)
        
    * **count**: The size to initialize (in bytes)
        
    
    ```cpp
    int *d_array;
    size_t size = 10 * sizeof(int);
    cudaMalloc((void**)&d_array, size);
    
    // Initialize device memory to 0
    cudaMemset(d_array, 0, size);
    ```
    

---

## (d) Error Code Check – `cudaGetErrorName()`

When a CUDA API function fails, it returns an error code of type `cudaError_t`.  
To convert this code into a human-readable string, `cudaGetErrorName()` is used.

* ```cpp
      const char* cudaGetErrorName(cudaError_t error);
    ```
    
* ```cpp
    int *d_array;
    cudaError_t err = cudaMalloc((void**)&d_array, -1); // intentional error
    
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorName(err));
    }
    ```
    

```cpp
Error: cudaErrorInvalidValue
```

You can see from the prototype of `cudaGetErrorName()` that it can be used both in host code and in device code.

```cpp
__host__ __device__ const char* cudaGetErrorName(cudaError_t error);
```

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    int *d_array = NULL;   // Pointer to device memory
    int h_array[N];        // Host memory for verification

    // (1) Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_array, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }

    // (2) Initialize device memory with zeros
    cudaMemset(d_array, 0, size);

    // (3) Copy data back from device to host for verification
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    printf("First %d elements after cudaMemset:\n", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // (4) Free device memory
    cudaFree(d_array);

    // (5) Intentional error: request invalid memory size
    int *d_bad = NULL;
    err = cudaMalloc((void**)&d_bad, (size_t)-1);
    if (err != cudaSuccess) {
        printf("Intentional failure -> %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
    }

    return 0;
}
```

### CUDA Memory Management Example (with Runtime API)

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    int *d_array = NULL;   // Pointer to device memory
    int h_array[N];        // Host memory for verification

    // (1) Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_array, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
        return -1;
    }

    // (2) Initialize device memory with zeros
    cudaMemset(d_array, 0, size);

    // (3) Copy data back from device to host for verification
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    printf("First %d elements after cudaMemset:\n", N);
    for (int i = 0; i < N; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // (4) Free device memory
    cudaFree(d_array);

    // (5) Intentional error: request invalid memory size
    int *d_bad = NULL;
    err = cudaMalloc((void**)&d_bad, (size_t)-1);
    if (err != cudaSuccess) {
        printf("Intentional failure -> %s (%s)\n",
               cudaGetErrorName(err), cudaGetErrorString(err));
    }

    return 0;
}
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755688434587/690c2e5d-200d-4762-a1cd-d7a026fe6143.png align="center")

# 2.Host–Device Memory Data Copy API.

## (a) Device Memory Copy – `cudaMemcpy()`

CUDA 프로그램에서 **호스트 메모리 ↔ 디바이스 메모리** 간 데이터 전송은 필수적이다.  
이때 사용하는 API가 `cudaMemcpy()` 이다.

---

Function Prototype

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```

* **dst**: destination pointer (copy target)
    
* **src**: source pointer (copy source)
    
* **count**: number of bytes to copy
    
* **kind**: copy direction, one of:
    
    * `cudaMemcpyHostToDevice` : Host → Device
        
    * `cudaMemcpyDeviceToHost` : Device → Host
        
    * `cudaMemcpyDeviceToDevice` : Device → Device
        
    * `cudaMemcpyHostToHost` : Host → Host (rarely used)
        

---

Example

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    const int N = 5;
    int h_array[N] = {1, 2, 3, 4, 5};  // Host array
    int h_result[N];                   // Host array for results
    int *d_array;                      // Device pointer

    size_t size = N * sizeof(int);

    // (1) Allocate device memory
    cudaMalloc((void**)&d_array, size);

    // (2) Copy data from Host → Device
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // (3) Copy back from Device → Host
    cudaMemcpy(h_result, d_array, size, cudaMemcpyDeviceToHost);

    // (4) Print result
    printf("Copied data:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_result[i]);
    }
    printf("\n");

    // (5) Free device memory
    cudaFree(d_array);

    return 0;
}
```

### `cudaMemcpyKind` Enum Items

```cpp
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost   = 0, // Copy from host memory to host memory
    cudaMemcpyHostToDevice = 1, // Copy from host memory (CPU) to device memory (GPU)
    cudaMemcpyDeviceToHost = 2, // Copy from device memory (GPU) to host memory (CPU)
    cudaMemcpyDeviceToDevice = 3, // Copy from one device memory location to another (same GPU)
    cudaMemcpyDefault = 4 // Automatically determine the direction based on pointer types
} cudaMemcpyKind;
```

Execution Result

```cpp
Copied data:
1 2 3 4 5
```