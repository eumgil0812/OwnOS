---
title: "Synchronization"
datePublished: Sat Aug 23 2025 11:27:04 GMT+0000 (Coordinated Universal Time)
cuid: cmeo6crj0000302lafqomgab2
slug: synchronization
tags: cuda

---

## 1\. Synchronization

Since a GPU executes thousands of threads simultaneously, **synchronization is essential** to ensure data consistency and to control execution order. CUDA provides synchronization mechanisms at various levels.

### (a) Synchronization Functions

**① Block-level synchronization**

* `__syncthreads()`
    
* All threads within the same block must reach this point before proceeding.
    
* Essential for cooperative computations using shared memory (e.g., tiled matrix multiplication).
    
* Does not affect threads outside the block.
    

**② Warp-level synchronization**

* `__syncwarp(mask)`
    
* Synchronizes threads within the same warp (32 threads).
    
* Useful when you want to synchronize only a subset of threads, not the entire block.
    
* On modern architectures, explicit warp sync may be required after warp divergence.
    

**③ Grid (kernel)-level synchronization**

* It is **not possible to synchronize across blocks within a single kernel**.
    
* For grid-level synchronization, the kernel must be spl
    
    * ex) `kernel1<<<...>>>(); cudaDeviceSynchronize(); kernel2<<<...>>>();`
        
* Cooperative Groups provide grid-level synchronization, but with certain limitations.
    

---

### (b) Atomic Functions

* Used to prevent data races when multiple threads update the same memory location simultaneously.
    
* Examples: `atomicAdd()`, `atomicSub()`, `atomicMax()`, etc.
    
* Atomic operations serialize access and can reduce performance.
    
* However, they are safer than race conditions caused by unsynchronized access.
    

---

### (c) Manual Control

Sometimes, explicit synchronization from the **CPU side** is required:

* `cudaDeviceSynchronize()` → blocks the CPU until **all GPU work** is finished.
    
* `cudaStreamSynchronize(stream)` → blocks the CPU until the specified **stream** finishes.
    

These are useful when mixing multiple kernels/streams and enforcing a specific order of execution.

---

### (d) Things to Watch Out For

* **Excessive synchronization causes performance loss** → minimize when possible.
    
* **Improper block sync** → threads outside a block must not call `__syncthreads()` (deadlock risk).
    
* **Atomic operations may become bottlenecks** → avoid using them across very large arrays.
    
* **Memory visibility** → without synchronization, writes to shared/global memory may not be visible to other threads in time.
    

## 2\. CUDA Streams and Concurrent Execution

### 1) Definition and Characteristics of CUDA Streams

* A **CUDA stream** is a **queue** that defines the order of GPU operations (kernel launches, memory copies, etc.).
    

**Characteristics**:

* Within the **same stream**, operations are executed sequentially in the order they were issued.
    
* Across **different streams**, operations can run concurrently (in parallel).
    
* By default, all CUDA calls go into the **default stream (stream 0)**.
    

👉 Using streams enables advanced optimizations such as overlapping **computation and data transfer**, and running **multiple kernels concurrently**.

### (a) Creating and Destroying Non-NULL Streams

* **Non-NULL streams** are streams explicitly created by the programmer (unlike the default stream, which is implicitly provided).
    
* They allow you to manage multiple independent execution queues on the GPU.
    

### (b) Specifying a Stream When Launching Kernels

```cpp
kernel<<<gridDim, blockDim, 0, stream>>>(args...);
```

The last parameter in the kernel launch configuration is the stream.

```cpp
myKernel<<<grid, block, 0, stream1>>>(dA, dB, dC);
myKernel<<<grid, block, 0, stream2>>>(dX, dY, dZ);
```

* you want me to also add a **timeline-style example** (like “Stream1: H2D → Kernel → D2H” overlapping with “Stream2”)?
    
* That could make this clearer for beginners.
    

### 2) Concurrent Execution of CUDA Operations

To fully utilize the GPU, it’s important to **overlap computation (kernel execution) with data transfers (Host ↔ Device copies)**.  
The key concepts here are **asynchronous memory copies** and **pinned memory**.

---

### (a) Asynchronous Memory Copy and Pinned Memory

* **Synchronous copy (**`cudaMemcpy`)  
    → The CPU must **wait until the copy is finished** before continuing.
    
* **Asynchronous copy (**`cudaMemcpyAsync`)  
    → The CPU can **issue the copy request and immediately move on**.  
    → This allows memory transfers and GPU kernel execution to happen **at the same time**.
    

⚠️ **Important condition:** The host memory must be **pinned (page-locked)** for true asynchronous transfers.

---

### Why is pinned memory necessary?

* **Pageable memory (default)** → the OS can move it around in RAM at any time.
    
    * The GPU cannot directly access it with DMA.
        
    * CUDA has to use a hidden **staging buffer**, adding overhead → slower and effectively synchronous.
        
* **Pinned memory (page-locked)** → fixed in physical RAM, cannot be moved by the OS.
    
    * The GPU can directly access it using DMA.
        
    * This makes transfers **faster** and allows **true asynchronous overlap** with kernels.
        

---

👉 **Summary:**  
To use asynchronous copies effectively, you **must use pinned memory**.

---

### (b) Allocating and Freeing Pinned Memory

CUDA makes it very simple to work with pinned memory:

```cpp
float* hA;
cudaMallocHost((void**)&hA, size * sizeof(float)); // 핀드 메모리 할당
// ... hA를 CPU 배열처럼 사용 가능 ...
cudaFreeHost(hA); // 해제
```

* `cudaMallocHost` → allocates pinned memory (page-locked), which the GPU can access directly.
    
* `cudaFreeHost` → frees the pinned memory.
    

### 3) Stream Synchronization

CUDA streams are **asynchronous by default**.  
This means when the CPU launches a kernel or a memory copy into a stream, it immediately moves on to the **next line of code**, while the GPU continues execution in the background.

👉 However, sometimes you need control, such as:

* “This kernel must finish before the next one starts.”
    
* “The CPU must wait for the GPU result before proceeding.”
    

In these cases, you use **synchronization**.

① Synchronizing a Specific Stream

```cpp
cudaStreamSynchronize(stream);
```

* The CPU will **wait until all operations in the specified stream are finished**.
    
* Example: when a kernel in `stream1` must complete before the CPU reads back its results.
    

② Event-based Synchronization

```cpp
cudaEvent_t event;
cudaEventCreate(&event);

cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0); // stream2는 event까지 기다림
```

* `stream1` records an event.
    
* `stream2` will not start until the event has fired.
    
    👉 This creates **ordering dependencies between streams** without forcing a global device-wide sync.
    

③ Device-wide Synchronization

```cpp
cudaDeviceSynchronize();
```

* The CPU waits until **all streams and all GPU work** are finished.
    
* Very powerful but costly — it stalls the entire GPU, so it should only be used when absolutely necessary.
    

## Example: Asynchronous Copy + Kernel Execution with Two Streams and Synchronization

**Key points**

* Use `cudaMallocHost` to allocate **pinned (page-locked) memory** → required for `cudaMemcpyAsync` to work as truly asynchronous.
    
* Create **two streams** and alternate between them like **double-buffering**.
    
* Use `cudaEventRecord` / `cudaStreamWaitEvent` to enforce the correct order of **copy → kernel → copy** inside each stream.
    
* Compare performance:
    
    * **(A) Single stream (sequential)** vs.
        
    * **(B) Dual streams (overlapped execution)** → measure time difference.
        

```cpp
// stream_overlap_demo.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void vecAdd(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ C,
                       int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

static void initArray(float* p, int n, float v) {
    for (int i = 0; i < n; ++i) p[i] = v;
}

int main(int argc, char** argv) {
    // 전체 원소 수 (기본: 4M)
    int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 22); // 4,194,304
    // 청크 크기 (기본: 1M)
    int CHUNK = (argc > 2) ? std::atoi(argv[2]) : (1 << 20);

    if (CHUNK > N) CHUNK = N;
    int numChunks = (N + CHUNK - 1) / CHUNK;

    size_t bytesAll = size_t(N) * sizeof(float);
    size_t bytesChunk = size_t(CHUNK) * sizeof(float);
    printf("N=%d (%.2f MB/array), CHUNK=%d (%.2f MB)\n",
           N, bytesAll / (1024.0*1024.0), CHUNK, bytesChunk / (1024.0*1024.0));

    // 핀드(페이지 잠금) 호스트 메모리
    float *hA, *hB, *hC;
    CHECK(cudaMallocHost(&hA, bytesAll));
    CHECK(cudaMallocHost(&hB, bytesAll));
    CHECK(cudaMallocHost(&hC, bytesAll));
    initArray(hA, N, 1.0f);
    initArray(hB, N, 2.0f);

    // 디바이스 버퍼 2세트 (더블버퍼링처럼)
    float *dA[2], *dB[2], *dC[2];
    for (int s = 0; s < 2; ++s) {
        CHECK(cudaMalloc(&dA[s], bytesChunk));
        CHECK(cudaMalloc(&dB[s], bytesChunk));
        CHECK(cudaMalloc(&dC[s], bytesChunk));
    }

    // 스트림 2개
    cudaStream_t stream[2];
    CHECK(cudaStreamCreate(&stream[0]));
    CHECK(cudaStreamCreate(&stream[1]));

    // 이벤트: 복사 완료/커널 완료 등을 스트림 내부 순서 제어에 사용(필수는 아님, 학습용)
    cudaEvent_t h2dDone[2], kernelDone[2];
    for (int s = 0; s < 2; ++s) {
        CHECK(cudaEventCreate(&h2dDone[s]));
        CHECK(cudaEventCreate(&kernelDone[s]));
    }

    dim3 block(256);
    // (청크 단위 그리드 크기는 매 반복에서 계산)

    // ------------------------------
    // (A) 단일 스트림 직렬 파이프라인
    // ------------------------------
    cudaStream_t serial;
    CHECK(cudaStreamCreate(&serial));

    cudaEvent_t tA_start, tA_stop;
    CHECK(cudaEventCreate(&tA_start));
    CHECK(cudaEventCreate(&tA_stop));

    CHECK(cudaEventRecord(tA_start, serial));
    for (int c = 0; c < numChunks; ++c) {
        int off = c * CHUNK;
        int count = (c == numChunks - 1) ? (N - off) : CHUNK;
        size_t bytes = size_t(count) * sizeof(float);
        int grid = (count + block.x - 1) / block.x;

        // H2D (동기/비동기 상관없이 같은 스트림에선 순차)
        CHECK(cudaMemcpyAsync(dA[0], hA + off, bytes, cudaMemcpyHostToDevice, serial));
        CHECK(cudaMemcpyAsync(dB[0], hB + off, bytes, cudaMemcpyHostToDevice, serial));

        // Kernel
        vecAdd<<<grid, block, 0, serial>>>(dA[0], dB[0], dC[0], count);

        // D2H
        CHECK(cudaMemcpyAsync(hC + off, dC[0], bytes, cudaMemcpyDeviceToHost, serial));
    }
    CHECK(cudaEventRecord(tA_stop, serial));
    CHECK(cudaEventSynchronize(tA_stop));
    float msA = 0.f;
    CHECK(cudaEventElapsedTime(&msA, tA_start, tA_stop));
    CHECK(cudaStreamDestroy(serial));
    CHECK(cudaEventDestroy(tA_start));
    CHECK(cudaEventDestroy(tA_stop));

    // ------------------------------
    // (B) 이중 스트림 겹침 파이프라인
    // ------------------------------
    cudaEvent_t tB_start, tB_stop;
    CHECK(cudaEventCreate(&tB_start));
    CHECK(cudaEventCreate(&tB_stop));

    CHECK(cudaEventRecord(tB_start, 0));
    for (int c = 0; c < numChunks; ++c) {
        int s = c & 1;         // 0,1,0,1...
        int off = c * CHUNK;
        int count = (c == numChunks - 1) ? (N - off) : CHUNK;
        size_t bytes = size_t(count) * sizeof(float);
        int grid = (count + block.x - 1) / block.x;

        // 1) H2D (A,B)
        CHECK(cudaMemcpyAsync(dA[s], hA + off, bytes, cudaMemcpyHostToDevice, stream[s]));
        CHECK(cudaMemcpyAsync(dB[s], hB + off, bytes, cudaMemcpyHostToDevice, stream[s]));
        // H2D 완료 시점 표시
        CHECK(cudaEventRecord(h2dDone[s], stream[s]));

        // 2) 커널은 H2D 완료 후에만 실행
        CHECK(cudaStreamWaitEvent(stream[s], h2dDone[s], 0));
        vecAdd<<<grid, block, 0, stream[s]>>>(dA[s], dB[s], dC[s], count);
        CHECK(cudaEventRecord(kernelDone[s], stream[s]));

        // 3) D2H는 커널 완료 후에만 실행
        CHECK(cudaStreamWaitEvent(stream[s], kernelDone[s], 0));
        CHECK(cudaMemcpyAsync(hC + off, dC[s], bytes, cudaMemcpyDeviceToHost, stream[s]));
        // 다음 반복에서 같은 s 버퍼를 재사용하기 전에, 이 스트림이 끝났는지 보장하려면
        // 루프 상단에서 wait를 더 둘 수도 있지만, 여기선 이벤트 체인으로 충분.
    }

    // 두 스트림의 모든 작업 완료 대기
    CHECK(cudaStreamSynchronize(stream[0]));
    CHECK(cudaStreamSynchronize(stream[1]));
    CHECK(cudaEventRecord(tB_stop, 0));
    CHECK(cudaEventSynchronize(tB_stop));
    float msB = 0.f;
    CHECK(cudaEventElapsedTime(&msB, tB_start, tB_stop));

    // 결과 검사(간단)
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (hC[i] != 3.0f) { ok = false; break; }
    }
    printf("Result check: %s\n", ok ? "OK" : "FAIL");

    // 성능 요약
    double GB = (3.0 * bytesAll) / (1024.0 * 1024.0 * 1024.0); // A,B 읽기 + C 쓰기(개념상)
    printf("[A] Single stream:   %8.3f ms,  Effective BW ~ %6.2f GB/s\n", msA, GB / (msA/1000.0));
    printf("[B] Dual streams:    %8.3f ms,  Effective BW ~ %6.2f GB/s\n", msB, GB / (msB/1000.0));
    printf("Speedup (A/B): %.2fx\n", msA / msB);

    // 정리
    for (int s = 0; s < 2; ++s) {
        CHECK(cudaFree(dA[s]));
        CHECK(cudaFree(dB[s]));
        CHECK(cudaFree(dC[s]));
        CHECK(cudaStreamDestroy(stream[s]));
        CHECK(cudaEventDestroy(h2dDone[s]));
        CHECK(cudaEventDestroy(kernelDone[s]));
    }
    CHECK(cudaFreeHost(hA));
    CHECK(cudaFreeHost(hB));
    CHECK(cudaFreeHost(hC));
    CHECK(cudaEventDestroy(tB_start));
    CHECK(cudaEventDestroy(tB_stop));
    return 0;
}
```