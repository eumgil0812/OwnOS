---
title: "CUDA Performance Measurement"
datePublished: Thu Aug 21 2025 13:22:46 GMT+0000 (Coordinated Universal Time)
cuid: cmelflurq000l02kzhm3a07t4
slug: cuda-performance-measurement
tags: cuda

---

GPUs are used for parallel processing to improve computational performance, meaning to shorten the time needed for the same operations.

Thus, when writing a CUDA program, its performance should be evaluated based on the execution time.

# **1\. Kernel Execution Time**

  
GPU computations are carried out through kernel launches. One of the key aspects in measuring the performance of a CUDA algorithm is the kernel execution time.

```cpp
    helloCUDA<<<1, 10>>>();
```

To measure GPU computation time, record the time before launching the kernel and after it finishes.

  
Since kernel launches return control to the host immediately, the host and device run asynchronously.

  
This means you need to wait until the kernel completes before stopping the timer. Otherwise, you’ll end up measuring a time close to zero.

  
Below is a sample code to measure kernel execution time.

```cpp
    helloCUDA<<<1, 10>>>();
    cudaDeviceSynchronize(); // Wait until all GPU tasks have completed
```

You might worry that because the host and device run asynchronously, a kernel could start running before data copying finishes.

Luckily, CUDA API calls are sequential by default.

  
All CUDA calls are placed into something called a *stream* (basically a queue). As long as your host code is just controlling device code, you don’t need to add extra synchronization.

  
But if you’re using multiple resources at once, you’ll need synchronization to keep the right order and exchange information properly.

---

# 2\. Data Transfer Time

When measuring the performance of a CUDA algorithm, it’s not only the GPU computation time that matters — data transfer time is important as well.  
A typical CUDA program consists of three steps:

1. copying data from the host (CPU) to the device (GPU),
    
2. performing computations on the GPU,
    
3. copying the results back from the device to the host.
    

The first and third steps don’t exist in a regular CPU-only program, and they introduce extra overhead. That’s why these transfer times also need to be included when evaluating performance.

```cpp
    // Copy from Host → Device

    //time check start
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    //time check finish
```

```cpp
    // Copy results back Device → Host

    //time check start
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    //time check finish
```

`cudaMemcpy()` runs synchronously with the host code. In other words, the host waits until the copy is finished before moving on.  
Because of this, you can measure the transfer time directly without having to call `cudaDeviceSynchronize()`.

---

# 3\. Performance Measurement and Analysis of a CUDA-Based Vector Addition Program

Let’s put this knowledge into practice and perform a simple timing experiment.

```cpp
// simple_vector.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

// ----------------- CUDA Kernel -----------------
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ----------------- CPU Reference -----------------
void vectorAddCPU(const int *A, const int *B, int *C, int N) {
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    int *h_ref = (int*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = i; h_B[i] = i * 10; }

    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Events for CUDA timing
    cudaEvent_t start, afterH2D, afterKernel, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterKernel);
    cudaEventCreate(&afterD2H);

    // ----------------- GPU Timing -----------------
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(afterH2D);

    vectorAdd<<<(N+255)/256, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(afterKernel);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(afterD2H);
    cudaEventSynchronize(afterD2H);

    float h2d_ms, kernel_ms, d2h_ms, total_ms;
    cudaEventElapsedTime(&h2d_ms, start, afterH2D);
    cudaEventElapsedTime(&kernel_ms, afterH2D, afterKernel);
    cudaEventElapsedTime(&d2h_ms, afterKernel, afterD2H);
    cudaEventElapsedTime(&total_ms, start, afterD2H);

    // ----------------- CPU Timing -----------------
    auto t1 = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_ref, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // -----------------result -----------------
    printf("**** Timer Report ****\n");
    printf("CUDA Total: %.5f ms\n", total_ms);
    printf("Computation (Kernel): %.5f ms\n", kernel_ms);
    printf("Data Trans. Host→Device: %.5f ms\n", h2d_ms);
    printf("Data Trans. Device→Host: %.5f ms\n", d2h_ms);
    printf("VecAdd on CPU: %.5f ms\n", cpu_ms);
    printf("**********************\n");

    // result check
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_ref[i]) { ok = false; break; }
    }
    printf("Check: %s\n", ok ? "PASS" : "FAIL");

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(afterH2D);
    cudaEventDestroy(afterKernel);
    cudaEventDestroy(afterD2H);

    return 0;
}

```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755781460148/a43bbfe5-5478-4b72-9c54-ddd0391f855d.png align="center")

## 🔑 **When** `cudaEventSynchronize()` Is Needed

**1\. Accurate timing (ensuring correct measurement)**  
When using `cudaEventElapsedTime()` to measure elapsed time, you need a final `cudaEventSynchronize()` to ensure that the event has actually finished.  
Without it, the GPU might still be running, and the measured time could end up close to zero.

**2\. When the host needs to use results immediately**  
For example, if the GPU is still copying data with `cudaMemcpyAsync` but the CPU needs to access that data right away, you must wait until the copy is complete.  
In this case, record an event and then call `cudaEventSynchronize()` to safely wait for completion.

**3\. Controlling the order of asynchronous tasks**  
When multiple streams or asynchronous operations are involved, you can enforce that certain host code only executes after a specific event has completed.  
This is essentially a way for the host to "wait on" a GPU event.

---

### 🔑 **When** `cudaEventSynchronize()` Is Not Necessary

**1\. Using synchronous APIs**  
The standard `cudaMemcpy` is blocking — the host already waits until the copy finishes — so you don’t need `cudaEventSynchronize()`.

**2\. After a kernel if** `cudaDeviceSynchronize()` is already used  
`cudaDeviceSynchronize()` waits until *all* GPU tasks are done, so you don’t need an additional event synchronization.  
However, this is less efficient because it stalls the entire device, not just the event you care about.

**3\. When the host doesn’t need to wait**  
If the kernel is launched and the host can continue doing other work without needing immediate results, you don’t need to synchronize right away.

---

### ✅ **Summary (with analogy)**

* `cudaDeviceSynchronize()` → *Wait until the entire GPU has finished* (host says: “I’ll wait for everything to complete”).
    
* `cudaEventSynchronize(event)` → *Wait until this specific event (copy, kernel, etc.) has finished* (host says: “Tell me when this part is done”).
    
* Synchronous functions like `cudaMemcpy` → already blocking, so no extra synchronization is needed.
    

## ⚡ GPU Isn’t Always the Winner

As you might have guessed from the title — **the answer is no.**  
For small input sizes, the overhead of data transfers dominates, and the CPU can actually finish faster.  
Take a look at the following example.

```cpp
// compare_cpu_gpu_vector_add.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

// ---------------- CUDA Kernel ----------------
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// ---------------- CPU Reference ----------------
void vectorAddCPU(const int *A, const int *B, int *C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

// ---------------- Benchmark ----------------
void benchmark(int N) {
    size_t size = N * sizeof(int);

    // Host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);
    int *h_ref = (int*)malloc(size);
    for (int i = 0; i < N; i++) { h_A[i] = i; h_B[i] = i * 10; }

    // Device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // CUDA events
    cudaEvent_t start, afterH2D, afterKernel, afterD2H;
    cudaEventCreate(&start);
    cudaEventCreate(&afterH2D);
    cudaEventCreate(&afterKernel);
    cudaEventCreate(&afterD2H);

    // ---------------- GPU Timing ----------------
    cudaEventRecord(start);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(afterH2D);

    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaEventRecord(afterKernel);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(afterD2H);
    cudaEventSynchronize(afterD2H);

    float h2d_ms, kernel_ms, d2h_ms, total_ms;
    cudaEventElapsedTime(&h2d_ms, start, afterH2D);
    cudaEventElapsedTime(&kernel_ms, afterH2D, afterKernel);
    cudaEventElapsedTime(&d2h_ms, afterKernel, afterD2H);
    cudaEventElapsedTime(&total_ms, start, afterD2H);

    // ---------------- CPU Timing ----------------
    auto t1 = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_ref, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // ---------------- Print Results ----------------
    printf("N=%d\n", N);
    printf("  CPU      : %.5f ms\n", cpu_ms);
    printf("  GPU Total: %.5f ms (H2D %.5f + Kernel %.5f + D2H %.5f)\n\n",
           total_ms, h2d_ms, kernel_ms, d2h_ms);

    // Validate correctness (optional)
    bool ok = true;
    for (int i = 0; i < 10; i++) {
        if (h_C[i] != h_ref[i]) { ok = false; break; }
    }
    if (!ok) printf("  ❌ Mismatch in results!\n");

    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_ref);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(afterH2D);
    cudaEventDestroy(afterKernel);
    cudaEventDestroy(afterD2H);
}

int main() {
    int test_sizes[] = {1024, 10000, 1000000, 10000000};
    for (int N : test_sizes) {
        benchmark(N);
    }
    return 0;
}
```

Result

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755782300181/bf9c4aaf-4540-4dad-b511-b3c7a9c37ea2.png align="center")

The reason is :

* **Small N favors the CPU**  
    For small input sizes, data transfer overhead dominates. The CPU can finish faster because it doesn’t need to copy data back and forth.
    
* **Kernels are fast but memory-bound**  
    Vector addition only performs one arithmetic operation per element, so performance is limited more by memory bandwidth than raw compute power.
    
* **Always consider total time (H2D + Kernel + D2H)**  
    Looking only at kernel time can be misleading. In real applications, data transfers are always part of the cost.
    

---