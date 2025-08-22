---
title: "Thread Layout & Indexing"
datePublished: Fri Aug 22 2025 11:37:51 GMT+0000 (Coordinated Universal Time)
cuid: cmemraski000002lba0sre36g
slug: thread-layout-and-indexing
tags: cuda, indexing, cudaindexing

---

**Thread layout** in CUDA refers to how threads are structurally arranged in a program (the geometric placement of threads across blocks and grids).  
**Indexing** refers to how each thread determines the exact piece of data it should work on.

When you first start CUDA programming, one of the most confusing parts is understanding how threads are laid out and how their indices are assigned. Built-in variables like `threadIdx` and `blockIdx` can feel unfamiliar and tricky at first.

To master parallel programming, you must understand both the layout rules and the indexing calculation.

## Example: Summing a Vector Larger Than 1024 Elements

On NVIDIA GPUs, the maximum number of threads per block is **1024**.  
This means the following constraint always applies:

```cpp
blockDim.x * blockDim.y * blockDim.z ≤ 1024
```

For instance:

* A vector of **2000 elements** cannot fit into a single block (1024 limit).
    
* Therefore, you must use multiple blocks arranged in a grid.
    

## (1) Deciding the Thread Layout

The layout of threads depends on the **grid** and **block dimensions**.

* Typically, you first choose the **block size** (based on performance and hardware constraints).
    
* Then, given the dataset size and block size, you compute the **grid size**.
    

---

## (2) Computing the Global Thread Index

Each thread must have a unique **global thread ID**, so it knows which data element to process.  
CUDA calculates this global ID using both the **block index** and the **thread index within the block**.

### 1D Grid (simplest case)

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Example: blockDim.x = 256, blockIdx.x = 2, threadIdx.x = 10 → idx = 522 → This thread handles the 522nd element of the vector.

### 2D Grid / 2D Block

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x
        + (threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x);
```

Here, the indices are “flattened” in row-major order: **x → y → z**.

## (3) Applying Indexing in a Kernel

Suppose `N = 2000` elements, `blockSize = 256`:

```cpp
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  // boundary check
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 2000;
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize; // ceil(N / blockSize)

    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
}
```

## (4) General Formula for 3D Grids and Blocks

To generalize to 3D:

```cpp
__device__ int globalThreadId3D() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;

    return x 
         + y * (gridDim.x * blockDim.x)
         + z * (gridDim.x * blockDim.x * gridDim.y * blockDim.y);
}
```

* **X direction**  
    `threadIdx.x + blockIdx.x * blockDim.x`  
    → Position inside the block + all threads in earlier x-blocks
    
* **Y direction**  
    `(threadIdx.y + blockIdx.y * blockDim.y) * (gridDim.x * blockDim.x)`  
    → How far up in y, scaled by the total length in x
    
* **Z direction**  
    `(threadIdx.z + blockIdx.z * blockDim.z) * (gridDim.y * gridDim.x * blockDim.x * blockDim.y)`  
    → How far up in z, scaled by the total size of an xy-plane