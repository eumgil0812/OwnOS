---
title: "Basic Parallel-Processing"
datePublished: Sun Aug 17 2025 12:42:16 GMT+0000 (Coordinated Universal Time)
cuid: cmefoedad000402lb4wwka4w6
slug: basic-parallel-processing

---

# 1\. Parallel-Processing Hardware ✨

## Flynn’s Taxonomy 🧭

Classifies architectures by **instruction streams** and **data streams**.

|  | **Single Data** | **Multiple Data** |
| --- | --- | --- |
| **Single Instruction** | **SISD** — one instruction over one datum (classic single-core, strictly serial) | **SIMD** — one instruction applied to many elements in lockstep (vector/array processors, CPU SIMD) |
| **Multiple Instruction** | **MISD** — many different instructions on the same data (largely conceptual/rare) | **MIMD** — many independent instruction streams over many data (multicore CPUs, clusters) |

* **SISD** 🧩: simplest form; serial execution on a single core.
    
* **MISD** 🧪: mostly theoretical in general-purpose computing.
    
* **MIMD** 🧠🧠: “many SISDs together”; each processor has **its own control unit & context**. Often used for **task-level parallelism (TLP)**.
    
* **SIMD** 🧮: **one common instruction** operates on **many data elements** in lockstep (vector/array processing).
    
* ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755434263956/9c98ee04-aa0d-404c-b8ab-abfa4df97d33.png align="center")
    

---

## Memory Models

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755434277468/53af6b92-a709-44bb-aa22-8cca57ed7a10.png align="center")

### (a) Shared-Memory Systems 🧠

Multiple compute units operate within **one coherent address space**.

* Example: **multicore CPUs** sharing main memory.
    
* Within a **GPU device**, all SMs can access **device global memory**, which behaves “shared” at the device level.
    

### (b) Distributed-Memory Systems 🌐

Each compute unit/device has **its own private memory**; data exchange requires **explicit communication**.

* Examples: multi-node **clusters**; a typical PC with **CPU + discrete GPU** (CPU RAM vs. GPU device memory).
    
* ⚠️ Communication (network/PCIe/NVLink) is costly, so efficient algorithms **minimize and overlap transfers**.
    

---

## GPUs are **SIMT** 🎯

GPUs are often likened to SIMD, but are more precisely **SIMT (Single Instruction, Multiple Threads)**.

**SIMT characteristics**

* Threads are organized in groups (e.g., **warps**) and a **single control unit** issues the same instruction in **lockstep** to the group.
    
* **Each thread** retains **its own control context** (registers, predicates).
    
* **Control-flow divergence** within a group is allowed; divergent paths are **serialized**, reducing efficiency.
    

> TL;DR: GPUs execute like SIMD at the group level, but preserve **per-thread identity** and allow **divergence** — that’s SIMT.  
> Memory-model wise: inside the GPU, SMs share device memory; at the **host + discrete GPU** level, the system is effectively **distributed memory**.

---

## Quick Takeaways ✅

* **MIMD** → heterogeneous tasks & independent control flows (task parallelism).
    
* **SIMD/SIMT** → identical operations over large datasets (data parallelism).
    
* **Shared vs. Distributed** → one shared address space vs. explicit inter-device communication.
    
* **GPU = SIMT** → group lockstep, **thread-private context**, **branch divergence** matters.
    

# 2\. Comparing CPUs and GPUs ✨

## Why GPUs Emerged (and How They Evolved) 🧭

As display resolutions climbed and 3D scenes became more complex, the **graphics pipeline** (vertex → rasterization → shading) outgrew what a CPU could push in real time.  
Enter the **GPU**: a processor designed from day one for **massively parallel**, per-pixel/per-fragment math—millions of tiny, similar computations that can run side-by-side.

* **Birth motive:** accelerate the graphics pipeline, where each pixel/shader invocation is largely independent.
    
* **Resulting design bias:** maximize **throughput** (many simple ALUs in parallel) rather than single-thread **latency**.
    
* **Generalization:** as shader hardware matured, developers exposed it for compute (**GPGPU/CUDA**), turning that graphics-optimized parallel engine into a general data-parallel accelerator.
    

---

## CPU vs GPU — What’s the Real Difference? 🤜🤛

### A) Multicore CPU architecture 🧠

A CPU is a **general-purpose** engine built to run *diverse, branchy, latency-sensitive* code.

* **Fat per-core design:** each core takes lots of silicon for **branch prediction**, **out-of-order & speculative execution**, and a deep **cache hierarchy (L1/L2/L3)**.
    
* **Single-core strength:** far higher **single-thread performance** and low-latency response than a single GPU core.
    
* **Why so much control logic?** Typical programs have **irregular control flow** and **non-contiguous memory access**. CPUs spend die area to *hide* those unpredictabilities and keep pipelines busy.
    

### B) GPU architecture 🎨⚙️

A GPU was built to **accelerate the graphics pipeline**, then extended to general compute. It organizes many simple ALUs under lightweight control.

* **Control style:** one scheduler controls **groups of threads** (warps/wavefronts). Threads execute in lockstep (SIMT); divergence is allowed but hurts efficiency.
    
* **Cache ratio:** GPUs do have caches, but devote a **smaller fraction of die area** to them (per ALU) and far more to raw arithmetic units.
    
* **Memory system:** to feed thousands of lanes, GPUs pair with **very high-bandwidth memory**—e.g., **GDDR6/GDDR6X** or **HBM(2e/3)**.
    
    * **Trade-off:** bandwidth ↑, **capacity per device often smaller** than typical system RAM. (Data-center GPUs can be large, but the principle stands: GPUs prioritize bandwidth over capacity.)
        
* **Throughput philosophy:** hide memory latency by **running massive numbers of threads** and swapping warps when one stalls.
    

---

## CPU vs GPU — Side-by-Side 🗂️

| Dimension | CPU 🧠 | GPU 🎨 |
| --- | --- | --- |
| Primary goal | **Low latency**, fast single-thread | **High throughput**, many threads in parallel |
| Core design | Few, **complex** cores (OoO, speculation, big caches) | Many, **simple** ALUs grouped under lightweight control |
| Control flow | Strong **branch prediction**, excels at irregular code | **SIMT**; divergence serializes paths (efficiency loss) |
| Memory hierarchy | Large private caches (L1/L2 per core, big shared L3) | Smaller per-lane caching; big **L2**, on-chip **shared memory** per block |
| Memory bandwidth | Moderate; relies on caches & prefetch | **Very high** (GDDR/HBM) to feed thousands of lanes |
| Best at | OS, apps, RPC handlers, mixed/branchy workloads | Pixels/shaders, dense linear algebra, ML, regular data-parallel kernels |
| Programming mindset | Task-/MIMD-oriented; branch-tolerant | Data-parallel/SIMT; coalesced access, minimize divergence, keep occupancy high |

# 3\. Amdahl vs. Gustafson — ultra-brief ⚡

| View | **Amdahl’s Law** | **Gustafson’s Law** |
| --- | --- | --- |
| What is fixed? | **Problem size** (strong scaling) | **Time budget** (weak scaling) |
| Question | “How much **faster** can we finish the same job?” | “How much **more work** can we do in the same time?” |
| Bottleneck lens | **Serial fraction** caps speedup | **Serial fraction** erodes near-linear scaling |
| Typical use | Low-latency, real-time, fixed-size benchmarks | High-throughput training/simulation, scaled workloads |