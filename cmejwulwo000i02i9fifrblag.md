---
title: "Vector Addition with CUDA"
datePublished: Wed Aug 20 2025 11:49:55 GMT+0000 (Coordinated Universal Time)
cuid: cmejwulwo000i02i9fifrblag
slug: vector-addition-with-cuda
tags: cuda

---

## 🚀 Learning Vector Addition with CUDA

### 1\. What is Vector Addition?

Vector addition means **adding two vectors of the same size element by element**.  
For example:

A=\[1,2,3\],B=\[4,5,6\]A = \[1, 2, 3\], \\quad B = \[4, 5, 6\]

then,

C=A+B=\[5,7,9\]C = A + B = \[5, 7, 9\]

---

### 2\. Why is Vector Addition Important?

* **Fundamental operation in science and engineering**  
    Physics simulations, machine learning, graphics, and signal processing all rely on repeated “vector + vector” operations.
    
* **Perfect example for parallel computing**  
    Each element addition is independent, making it ideal for GPUs where thousands of threads run in parallel.
    
* **Classic CUDA introductory example**  
    Learning CUDA starts with understanding the **relationship between host (CPU) and device (GPU) memory**. Vector addition is the simplest way to demonstrate data transfer, parallel execution, and result retrieval.
    

---

### 3\. CUDA Vector Addition Example Code

```cpp
//simple_vector.cu
#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel: each thread adds one element from the vectors
__global__ void vectorAdd(const int *A, const int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(int);

    // ----- Host memory (CPU) -----
    int h_A[10], h_B[10], h_C[10];
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 10;
    }

    // ----- Device memory (GPU) -----
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy from Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel (threads = N)
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy results back Device → Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755690357716/e8193b7e-69a5-44ff-95aa-b812319a8bb2.png align="center")

1. **Host (CPU memory):** h\_A, h\_B, h\_C
    
2. **Device (GPU memory):** d\_A, d\_B, d\_C
    
3. **Data flow in CUDA:**
    
    * Host → Device: transfer input data
        
    * Device executes the kernel in parallel
        
    * Device → Host: transfer back the result
        

---

## 🔍 Understanding `int i = blockIdx.x * blockDim.x + threadIdx.x;`

In CUDA, this is the formula each thread uses to determine the index of the data it should process.  
Since the GPU executes thousands of threads simultaneously, each thread needs to know *“which element am I responsible for?”*

---

### 1\. Meaning of Variables

**threadIdx.x**  
→ The thread’s index **within a block** (starting from 0)  
→ *“Which thread am I inside my block?”*

**blockIdx.x**  
→ The block’s index **within the grid** (starting from 0)  
→ *“Which block do I belong to in the grid?”*

**blockDim.x**  
→ The number of threads contained in a block  
→ *“What is the size of my block (i.e., how many threads are in one row)?”*

---

### 2\. Meaning of the Formula

```cpp
i = blockIdx.x * blockDim.x + threadIdx.x
```

👉 In other words,

* `blockIdx.x * blockDim.x` → The **starting global index** of my block.
    
* `+ threadIdx.x` → My **position inside the block**.
    

📌 As a result, `i` represents the **global thread index in the entire grid**.

---

### Example:

Suppose `N = 10`, with **2 blocks** and **5 threads per block**:

* Block size (`blockDim.x`) = 5
    
* Thread index (`threadIdx.x`) = 0 ~ 4
    
* Block index (`blockIdx.x`) = 0, 1
    

**Calculation:**

* **Block 0** → `i = 0 * 5 + threadIdx.x = 0, 1, 2, 3, 4`
    
* **Block 1** → `i = 1 * 5 + threadIdx.x = 5, 6, 7, 8, 9`
    

➡️ Each thread gets a **unique ID** (0–9), ensuring every element of the vector is processed exactly once.