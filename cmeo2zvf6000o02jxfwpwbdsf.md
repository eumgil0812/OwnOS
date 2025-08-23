---
title: "Memory Access Performance Optimization"
datePublished: Sat Aug 23 2025 09:53:03 GMT+0000 (Coordinated Universal Time)
cuid: cmeo2zvf6000o02jxfwpwbdsf
slug: memory-access-performance-optimization
tags: cuda

---

### In large-scale GPU programs, many threads access memory at the same time, so performance is often limited by **memory bandwidth**.

Therefore, improving **memory access efficiency** and maximizing bandwidth is one of the key factors for speeding up a k

# 1\. Global memory access optimization

## 1) Aligned and **coalesced** memory access

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755938497573/e2498597-69e0-4232-b7d6-3edc4b8d3544.png align="center")

**What are L1 / L2?**

* **L1 cache**: A small, very fast cache **attached to each SM**. It captures values a warp has just read or written.
    
* **L2 cache**: A **large cache shared** by the entire GPU. All SMs funnel through it.
    
* **DRAM**: The largest but **slowest** global memory.
    

**Why is “32 bytes” important?**

* Global-memory (→ L2) transactions typically use a **32-byte sector** as the **minimum transfer unit**.
    
* How many **32B sectors** a warp (32 threads) touches for a load/store determines the **number of transactions** (i.e., bandwidth consumption).
    
* You can think of L1 as having **128-byte lines** (= 4 × 32B sectors). It pulls in a line (or sectors) and feeds registers.
    

**Intuition**  
If a warp’s addresses are **nicely packed/contiguous**, only a **few 32B sectors** are touched. If the addresses are scattered, many sectors must be fetched → slower.

**Rule of thumb (coalescing)**  
When a warp’s 32 threads read `float` (4B) from **contiguous addresses**, the access spans exactly **128B (= 32 × 4B)**, so a **single 128B block** covers it → **maximally efficient**.

```cpp
// good:Coalesced Access
int tid = threadIdx.x + blockIdx.x * blockDim.x;
C[tid] = A[tid] + B[tid]; // 연속 접근 → 빠름

// bad:Strided Access
int tid = threadIdx.x + blockIdx.x * blockDim.x;
C[tid] = A[tid * stride] + B[tid * stride]; // 띄엄띄엄 접근 → 느림
```

**Bottom line:** On GPUs, **memory-access pattern optimization is performance**. Although L1/L2 caches exist, **global memory bandwidth is usually the bottleneck**, so **aligned, coalesced access is essential**.

## 2) Structure of Arrays (SoA) vs Array of Structures (AoS)

On GPUs, **how you lay out your data structures** can significantly affect **memory-access efficiency**.

### Array of Structures (AoS)

```cpp
struct Particle {
    float x, y, z;
    float velocity;
};
Particle particles[N]; // AoS
```

* **Pros:** Intuitive; familiar to CPU-style code.  
    **Cons:** Even if you only need `x`, the layout forces `y`, `z`, `velocity`, etc. to be loaded as well → wasted bandwidth / inefficient.
    

### Structure of Arrays(SoA)

```cpp
struct Particles {
    float x[N], y[N], z[N];
    float velocity[N];
};
Particles particles; // SoA
```

* **Pros:** Place the same field contiguously → enables **coalesced/aligned** access on GPUs.  
    **Cons:** Can hurt code readability/ergonomics.
    
    👉 On GPUs, **SoA is usually much faster than AoS**, especially when thousands of threads read only a specific attribute.
    

# 2) Shared Memory Access Optimization

#### 1) One-line concept

Shared memory is like a **vault split into 32 banks**.  
When a **warp (32 threads)** accesses it simultaneously, if threads hit **different banks**, the access completes **in one cycle**; if multiple threads hit the **same bank**, the access is **serialized** → slower (**bank conflict**).

**Exception:** If all threads read the **same address**, the hardware performs a **broadcast**, so there’s **no conflict** (fast).

---

## 2) Why does it happen? (address → bank mapping)

* For `float` (4 bytes) the bank is typically:
    
    ```cpp
    bank = (address / 4) % 32
    ```
    

---

## 3) Three core fixes

+1 padding:

1. ```cpp
    __shared__ float tile[BLOCK][BLOCK+1]; // ← stride를 33으로 틀어 충돌 회피
    ```
    
    Even for column reads, `(i*33 + j) % 32` differs per thread, so accesses are **spread across banks**.
    
    **Match the indexing direction:**  
    When writing/reading the shared array, make `threadIdx.x` the fastest-varying dimension (row).
    
    * Write: `tile[ty][tx] = ...;`
        
    * Read (MAC loop): `tile[ty][k]`, `tile[k][tx]` (**with padding**)
        
    
    **Leverage broadcast:**  
    If many threads must read the **same element**, arrange the access so they read the **same address** (broadcast) → **no bank conflict**.
    

---

## 4) Transpose

```cpp
// Bad: 충돌 가능 (BLOCK=32일 때 특히)
__shared__ float sA[32][32];
sA[ty][tx] = A[row*N + col];
__syncthreads();
B[col*M + row] = sA[tx][ty];   // ← 열로 읽어 32-way 충돌

// Good: +1 패딩으로 해결
__shared__ float sA2[32][33];
sA2[ty][tx] = A[row*N + col];
__syncthreads();
B[col*M + row] = sA2[tx][ty];  // stride=33 → 충돌 거의 없음
```

---

## 5) Simpler analogy

* 32 **bank counters** (banks) and 32 **customers** (threads) arrive at the same time.
    
* If everyone goes to **different counters**, it’s done **in one shot**.
    
* If they crowd the **same counter**, they must **line up and go one-by-one** → slower.
    
* If everyone reads the **same notice** (same address), it’s like a **PA broadcast (broadcast read)** → served **once** for all.
    
* **+1 padding** is the trick of **“shifting the counter numbers by one”** so people don’t collide at the same counter.
    

---

### 6) Practical checklist (memorize)

* For shared arrays, use `[T][T+1]` padding (especially in transpose and GEMM tiles).
    
* If there’s any **column (vertical) access**, consider padding **mandatory**.
    
* Check **“Shared Memory Bank Conflicts”** in Nsight Compute to verify.
    
* With `double` (8B), a single thread can span **two banks** (2-way conflict). Bad patterns make it worse → add padding / redesign access.
    

**One-line takeaway**  
**Bank conflict = simultaneous access to the same bank → serialization.**  
In most cases, **+1 padding** and the right **indexing direction** fix it.