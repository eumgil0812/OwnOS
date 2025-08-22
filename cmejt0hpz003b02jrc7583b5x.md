---
title: "Basic Flow of a CUDA Program"
datePublished: Wed Aug 20 2025 10:02:31 GMT+0000 (Coordinated Universal Time)
cuid: cmejt0hpz003b02jrc7583b5x
slug: basic-flow-of-a-cuda-program
tags: cuda

---

A CUDA program consists of **host code** and **device code**.

The CPU is the primary processing unit of a computer system, along with the operating system. To use other processing units such as the GPU, the host code must call the kernel.

The main memory space of a computer system is also managed by the CPU and is typically the DRAM installed on the motherboard. This system memory is often referred to as **main memory**, and from the perspective of CUDA programs, it is usually called **host memory**.

The CPU and GPU are independent devices, and each uses a separate memory space.

The GPU uses **device memory**, but all data is initially stored in host memory. Therefore, to process data with the GPU, the data in host memory must be copied to device memory.

Thus, the first step in the general flow of a CUDA program is to copy the input data from **host memory** to **device memory**.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755683448041/1443904c-36b5-46e3-b630-ce6ac1409ff2.png align="center")

The next step is **GPU computation**.  
GPU computation begins with a **kernel launch**, and all data is managed within **device memory**.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755683833545/f873acb3-db16-4931-90e5-938b3e0f2b56.png align="center")

Since the results of computations performed by the GPU through a kernel launch are stored in device memory, they must be transferred back to host memory. Therefore, the final step in the flow of a CUDA program is copying the result data from device memory to host memory.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755683858452/d1ea9657-7a5b-4796-b49b-b9aebef28d1c.png align="center")