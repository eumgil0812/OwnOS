---
title: "CUDA Execution Model"
datePublished: Sat Aug 23 2025 05:22:19 GMT+0000 (Coordinated Universal Time)
cuid: cmentbovx000802jo2es6bneu
slug: cuda-execution-model
tags: cuda

---

## NVIDIA GPU Architecture

In 2006, NVIDIA introduced CUDA, enabling parallel computation on GPUs rather than CPUs. Since then, GPUs have evolved from being just graphics processors into the **core engines of general-purpose computation**.

But here’s the big question:  
👉 *While CPUs only have a handful of cores, how can GPUs scale to thousands of them?*

To answer that, we need to take a closer look at the design philosophy behind **NVIDIA’s GPU architecture**.

### 1\. Streaming Multiprocessor (SM)

The image below shows the internal structure of a **Streaming Multiprocessor (SM)** in the Fermi architecture.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755923273757/e03053c9-501c-42f2-9d6f-6b176cdbe96a.png align="center")

Fermi Architecture: The **Fermi architecture** was NVIDIA’s first GPU design built with **general-purpose parallel computing (GPGPU)** in mind. It introduced key features such as a cache hierarchy, ECC (Error-Correcting Code) memory support, and even C/C++ compatibility.

Today, Fermi is no longer in use. Modern GPUs start at least from **Pascal (2016)** and, in the HPC world, have progressed through **Volta → Turing → Ampere → Hopper**. Still, Fermi remains historically significant as the first real step toward GPUs becoming full-fledged computing engines.

**SM** stands for *Streaming Multiprocessor*. A single GPU is composed of multiple SMs, each functioning as a building block for parallel execution.

Inside an SM, you’ll find not only multiple **CUDA Cores**, but also supporting structures such as:

* A **register file**
    
* **Shared memory**
    
* An **L1 cache**
    

The diagram above illustrates the internal organization of an SM in the Fermi architecture.

### 2\. CUDA Cores

A **CUDA Core** is the most fundamental computational unit of an NVIDIA GPU — the smallest unit that actually performs arithmetic operations.

Looking more closely, each CUDA Core contains specialized execution units such as:

* **FP (Floating Point) units** for floating-point operations
    
* **INT (Integer) units** for integer operations
    

In general, you can think of **one CUDA Core as the hardware engine that executes one GPU thread**.

---

### Comparing CUDA Cores with CPU Cores

* **CPU Core**: Designed with complex control logic, deep cache hierarchies, and high single-thread performance, but limited to only a few cores.
    
* **CUDA Core**: Much simpler and lighter in design, but replicated by the thousands, allowing massive parallel execution of workloads.
    

---

### Evolution Across Generations

As GPU architectures advanced, CUDA Cores evolved beyond just floating-point operations:

* **Turing** introduced the ability to execute **FP and INT operations in parallel**
    
* **Volta** added **Tensor Cores**, dedicated to accelerating matrix multiplications in AI and deep learning
    

---

### A Word of Caution

It’s misleading to assume that *“more CUDA Cores = more performance.”*  
Real-world performance depends heavily on factors such as clock speed, IPC (instructions per cycle), memory bandwidth, and overall architectural improvements.

## CUDA Thread Hierarchy and GPU Hardware

## 1\. Grid → GPU

A **grid** always executes on a **single GPU**.  
Even if a system has multiple GPUs, a single grid cannot span across them or migrate between GPUs during execution.

On the other hand, a single GPU can handle multiple grids.  
This means that a single GPU can run **several CUDA programs (kernel launches) concurrently**, sharing resources across them.

### 2\. Thread Block → SM

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755924458816/b260b20b-6dc8-4bbf-b719-97c1c5f2da6f.png align="center")

Basic Principle

* A **grid** is composed of multiple **thread blocks**.
    
* When a grid is assigned to a GPU, its **SMs (Streaming Multiprocessors)** divide up the work.
    
* In other words: **the unit that executes a thread block is the SM**.
    

---

Block Distribution

* When a block is created, the GPU scheduler assigns it to an available SM.
    
* Once assigned, the block stays **pinned** to that SM until execution finishes — it never migrates to another SM.
    
* With multiple SMs available, the scheduler typically distributes blocks evenly (e.g., round-robin style).
    

---

Active Blocks

* Each SM has a hardware-defined limit on how many blocks it can keep active at once.
    
* Example: in some architectures, an SM can host up to **8 active blocks simultaneously**.
    
* If there are more blocks than the SM can handle (or resources are insufficient), the extra blocks are queued and wait until resources free up.
    

### 3\. Warp & Threads → CUDA Cores inside an SM

What is a Warp?

* Threads inside a thread block are grouped into units of **32**, called a **warp**.
    
* A warp is the **smallest scheduling unit** in a GPU.
    
* All 32 threads in a warp execute the **same instruction simultaneously** — this is the **SIMT (Single Instruction, Multiple Thread)** model.
    

---

Relationship Between CUDA Cores and Warps

* Each thread is executed on a **CUDA Core**.
    
* For example, in the Fermi architecture, an SM contained 32 CUDA Cores — meaning the 32 threads of a warp could be mapped **one-to-one** across the 32 cores for true parallel execution.
    
* In short: **1 warp = 32 threads = up to 32 CUDA Cores running in parallel**.
    

---

Scheduling

* The **warp scheduler** selects which warp to dispatch and issues its instruction.
    
* All 32 threads in the chosen warp execute that instruction in parallel.
    
* If the number of CUDA Cores per SM is greater than 32, multiple warps can be executed at the same time.
    

---

Key Takeaways

* **Thread Block → multiple Warps**
    
* **Warp (32 threads) → executed on CUDA Cores inside an SM**
    
* **Scheduling is done at the warp level** under the SIMT model
    

### 4\. Thread Context

| Item | CPU | GPU |
| --- | --- | --- |
| Concurrent Threads | Dozens at most | Thousands to tens of thousands |
| Context Storage | Saved in memory by the OS (PCB, etc.) | Stored directly in SM register files |
| Context Switch Cost | High (requires save/restore) | Very low (already resident in hardware) |
| Execution Unit | Individual threads | Warps (32 threads) |

👉 **CPU analogy**: “Carrying a small bag — you need to go back to the storage room (OS memory) whenever you want to switch.”  
👉 **GPU analogy**: “Owning thousands of lockers (SM registers) — everything is already stored, ready to be pulled out instantly.”

---

### CPU vs GPU Context Switching

* On a **CPU**, the operating system must save a thread’s state into memory and reload it later. This makes context switching heavy and relatively slow. CPUs are therefore optimized for managing **a small number of powerful threads** efficiently.
    
* On a **GPU**, the contexts of thousands of threads are stored directly in the SM’s register files. The warp scheduler can instantly switch between warps with virtually no overhead. This allows GPUs to keep their compute units busy, even when some threads are stalled waiting for memory.
    

👉 In practice: **CPUs excel at complex logic and single-thread performance, while GPUs shine at large-scale parallelism.**

---

## 5\. Zero Context Switch Overhead

* **CPU context switching**: Requires saving/restoring registers and program counters → significant overhead.
    
* **GPU context switching**: All thread contexts are already resident in hardware (SM register file) → overhead is effectively **zero**.
    

**Result:** GPUs can juggle thousands of threads with rapid switching, using this ability to **hide memory latency** and keep execution units fully utilized.

## 6\. Warp Divergence

> “When I first encountered the concept of warp divergence, it felt like an orchestra where every musician was trying to play from a different score.  
> *If threads are supposed to be independent, why are they all getting stuck together?*  
> The GPU execution model demands a hardware-centric way of thinking that goes beyond pure software intuition.”  
> — *PSK. me!*

---

### What Is Warp Divergence?

In CUDA, threads each have their own independent **context** (registers, program counter, etc.).

However, the hardware executes instructions in groups of **32 threads**, called a **warp**. Under the **SIMT model** (Single Instruction, Multiple Threads), every thread in a warp must execute the **same instruction** at the same time.

The problem arises when threads in the same warp take **different control paths** (e.g., inside an `if/else`). This situation is known as **warp divergence**.

---

### Example Code

```cpp
__global__ void warp_divergence_example(int *output) {
    int tid = threadIdx.x;

    if (tid % 2 == 0) {
        // Even threads → Branch A
        output[tid] = 100;
    } else {
        // Odd threads → Branch B
        output[tid] = 200;
    }
}
```

Execution Process (per warp of 32 threads)

1. Even threads execute `output[tid] = 100;` → odd threads remain **idle**.
    
2. Odd threads execute `output[tid] = 200;` → even threads remain **idle**.
    
3. Both branches **reconverge**, and execution continues.
    

👉 In effect, a task that could have been finished **in one pass** now requires **two passes**, cutting performance in half.

---

### Why Is This a Problem?

Divergence reduces efficiency **in proportion to the number of branches**.

In the example above, there are two branches → warp efficiency ≈ **50%**.

Worst case: if all 32 threads in a warp diverge, efficiency can drop to **1/32**.

So instead of true parallelism, the GPU ends up doing **sequential execution with idle waiting**.

---

### Intuitive Analogy

A warp is like **32 musicians playing together in an orchestra**.

Normally, they all follow the same sheet of music to create harmony.  
But imagine half of them pull out Score A while the other half pull out Score B.

The orchestra has a strict rule: *“everyone must play the same sheet at the same time.”*

* First, the A group performs while the B group waits.
    
* Then, the B group performs while the A group waits.
    

As a result, the concert takes **twice as long** to finish.

---

### Key Takeaways

* **Warp = group of 32 threads**
    
* All threads in a warp must execute the **same instruction simultaneously**
    
* When divergence occurs → each branch is executed sequentially, while the other threads sit idle
    
* Warp divergence is a **major source of performance loss** and must be carefully considered when optimizing CUDA programs
    

## Hiding Memory Access Latency

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755925604314/5f227056-71f0-48a5-8dfc-bf0af7cc3cfd.png align="center")

### 1) What Is Memory Access Latency?

GPU operations can be divided into two broad categories:

* **Memory Access** – reading from or writing to global memory
    
* **Computation** – performing arithmetic on data stored in registers or caches
    

The problem is that whenever a memory access occurs, the compute cores (**CUDA Cores**) have to **wait idly** until the data arrives. This waiting period is called **memory access latency**.

On a CPU, such latency is mitigated with techniques like **high cache hit rates**, **branch prediction**, and **out-of-order execution**.  
But GPUs, with thousands of cores running in parallel, take a very different approach: instead of trying to reduce latency, they **hide it entirely**.

---

### 2) The GPU’s Solution: Massive Threads

GPUs keep **far more threads ready than the number of cores available**.

* When one warp stalls because of a memory access,
    
* Another warp immediately takes its place and continues computation.
    

By alternating execution like this, CUDA cores are rarely idle — computation keeps flowing even while some threads are waiting for memory.

👉 This strategy is known as **Latency Hiding**.

---

### 3) Visualizing the Concept

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755925745829/7c1244b8-d6e8-42dd-9462-2a0e9f084683.png align="center")

.

* * **Thread 1**: Stalls while waiting for a memory access → its CUDA Core would normally sit idle.
        
    * But if **Thread 2** is already ready to go → the SM immediately switches to it and starts computation.
        
    * Then **Thread 3** comes in, and execution continues seamlessly.
        
    
    👉 In other words: *“while one thread is waiting, another thread fills the gap.”*
    

---

### 4) Why Can GPUs Do This?

On a **CPU**, switching threads is expensive because the thread context must be **saved to and restored from memory**.

On a **GPU**, however, all thread contexts are stored directly in the **register file inside the SM**.

As a result, context switching overhead is effectively **zero (Zero Context Switch Overhead)**.

This is what enables GPUs to keep **thousands of threads resident and ready**, ensuring smooth, uninterrupted execution even when many threads are waiting on memory.

---

### 5) Algorithm Considerations

* **I/O-Bound Workloads (data access heavy)**  
    → Increasing the number of threads helps hide memory latency more effectively.  
    → Example: vector copies, simple dataset transformations
    
* **Compute-Bound Workloads (computation heavy)**  
    → Too many threads can actually hurt performance due to register pressure and resource contention.  
    → Example: complex math operations, deep learning matrix multiplications
    

A common rule of thumb is to start with about **10× as many threads as CUDA cores**, then refine based on profiling tools like **Nsight Compute**.

---

### 6) Key Takeaways

* **Memory access latency = wasted performance**
    
* GPUs hide this latency with **massive threading** and **zero-overhead context switching**
    
* For I/O-bound tasks: add more threads to mask latency
    
* For compute-bound tasks: too many threads can backfire
    
* **Basic strategy**: start with *“# of cores × 10” threads*, then fine-tune with profiling