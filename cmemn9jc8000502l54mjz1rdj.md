---
title: "CUDA Thread Hierarchy"
datePublished: Fri Aug 22 2025 09:44:54 GMT+0000 (Coordinated Universal Time)
cuid: cmemn9jc8000502l54mjz1rdj
slug: cuda-thread-hierarchy
tags: cuda, cudathread

---

## **1\. CUDA Thread Hierarchy**

The CUDA thread hierarchy consists of four levels: **threads**, **warps**, **blocks**, and **grids**.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755783097429/c9d1142c-6328-4c3b-bcdc-b0c92319726d.png align="center")

---

### **(a) Thread**

The smallest unit is the **thread**. A thread is the lowest level in the CUDA thread hierarchy and serves as the basic unit that performs computations or utilizes CUDA cores.

The kernel code you write is shared by all threads, but each thread executes the kernel code **independently**.

---

### **(b) Warp**

A **warp** is a group of 32 threads, and it is also the fundamental execution unit in CUDA.

Being the fundamental execution unit means that all threads within a warp are controlled by a single control unit. In the GPU’s SIMT (Single Instruction, Multiple Threads) architecture, the warp serves as the execution unit of multithreading. In other words, 32 threads execute the same instruction simultaneously, making the warp a crucial concept in CUDA programming.

---

### **(c) Block**

A **block** (or thread block) is a collection of warps.

Each block assigns a unique identifier (ID) to its threads. Importantly, within the same block, no two threads share the same thread ID. However, threads in different blocks may have the same thread ID.

Each block itself has a unique block ID. Therefore, to precisely identify a specific thread, both the block ID and the thread ID must be used together.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755783518619/6d23eb1e-8ba2-4fa0-942f-25497babf2dd.png align="center")

### **(d) Grid**

At the highest level of the CUDA thread hierarchy is the **grid**.  
A grid is a group of blocks that contains multiple blocks.  
Each block within a grid has its own unique **block ID**, which distinguishes it from other blocks in the same grid.

## **2\. Built-in Variables for the CUDA Thread Hierarchy**

In the CUDA thread hierarchy, grid blocks can be arranged in 1D, 2D, or 3D, and threads within a block can also be arranged in 1D, 2D, or 3D.  
For threads to determine which data they should process, each thread must know which block it belongs to, as well as its own thread index within that block.

To support this, CUDA provides **built-in variables**.  
These built-in variables allow each thread to identify the configuration of the current grid and block, as well as its own block ID and thread ID.

The values of these variables are assigned when the kernel is launched, and each thread can reference its own values. However, these built-in variables are **read-only** and cannot be modified within the code.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755853108497/86764c40-b6fe-42bf-b902-74748894b3ce.png align="center")

### **(a) gridDim**

`gridDim` is a built-in variable of type **structure** that contains information about the shape of the grid.

Its members `x`, `y`, and `z` represent the size of the first, second, and third dimensions of the grid, respectively.

For example, if the grid is defined as `(3, 2, 1)`, the unused dimension is represented as `1`.  
Thus, `gridDim.x`, `gridDim.y`, and `gridDim.z` would have the values `3`, `2`, and `1`, respectively.

The `gridDim` variable is **shared by all threads in the kernel**.

---

### **(b) blockIdx**

`blockIdx` is a built-in variable of type **structure** that stores the index of the block to which the current thread belongs.

Block indices start from `0`, and for unused dimensions, the index value is `0`.

The `blockIdx` variable is **shared by all threads within the same block**.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755853993152/26b24a00-e0a9-4cd4-8f64-d76d93925695.png align="center")

### **(c) blockDim**

`blockDim` is a built-in variable of type **structure** that contains information about the dimensions of a block.  
Each dimension length is a positive integer greater than or equal to 1.

When the kernel is launched, the grid and block configurations are determined, and all blocks within the same grid share the same dimensions.  
Therefore, the value of `blockDim` is the same across all threads in the grid.

---

### **(d) threadIdx**

`threadIdx` is a built-in variable of type **structure** that stores the **thread index** assigned to the current thread within its block.  
Each thread in a block has a unique thread index, and no two threads within the same block share the same `threadIdx`.

## 3\. Importance of Warps

### **1\. What is a warp?**

  
A warp is the unit of execution in which GPU threads are grouped and executed simultaneously.  
On NVIDIA GPUs, a warp consists of **32 threads**.

This means the GPU scheduler issues instructions **per warp**, not per individual thread.

👉 Put simply, while a CPU executes threads independently, on a GPU, 32 threads are always bundled together and execute the same instruction in lockstep.

---

### **2\. The meaning of “32 consecutive threads”**

  
Understanding how threads are numbered is important. In CUDA, thread IDs are assigned in the order:

```cpp
threadIdx.y=0, threadIdx.x=0~7  →  스레드 0~7
threadIdx.y=1, threadIdx.x=0~7  →  스레드 8~15
```

If the **z-dimension** is also used, the thread indices increase in the order **x → y → z**.

Therefore, “32 consecutive threads” refers to threads grouped in this numbering order, such as **0–31, 32–63, …** in sets of 32.  
Each of these groups of 32 threads forms a **warp**, and the GPU executes instructions on a **warp basis**.

---

### **3\. Why is this important?**

Because execution happens at the **warp level**:

* If threads within the same warp follow **different execution paths** (e.g., due to conditional branches), performance decreases. This phenomenon is called **warp divergence**.
    
* Memory access is also managed at the **warp level** through **memory coalescing**. If threads in a warp access **consecutive memory addresses**, performance is high. However, if memory accesses are scattered, performance drops significantly.