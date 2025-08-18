---
title: "Concurrency and Threads"
datePublished: Mon Aug 18 2025 09:45:43 GMT+0000 (Coordinated Universal Time)
cuid: cmegxj62d001v02judqgq1oz6
slug: concurrency-and-threads
tags: concurrency

---

### Understanding Threads in Operating Systems

Operating systems provide **abstractions** to make hardware easier to use and share:

* **CPU abstraction**: A single physical CPU is turned into multiple *virtual CPUs*, giving the illusion that many programs run at once.
    
* **Memory abstraction**: Each process is given its own *virtual address space*, so programs behave as if they have private memory, even though the OS is multiplexing physical memory (and sometimes disk).
    

Now, let’s look at a new abstraction within a single process: the **thread**.

---

### What is a Thread?

* A traditional process has a single **execution point** (a Program Counter, or PC).
    
* A **multi-threaded process** can have multiple execution points, each running independently.
    
* Threads are almost like separate processes, with one critical difference: **they share the same address space**.
    

This is why threads are often called **“lightweight processes.”**

---

### Thread State

Each thread maintains its own state:

* **Program Counter (PC)** – tracks the next instruction
    
* **Registers** – used for computation
    
* **Stack** – stores local variables, function arguments, return values, etc.
    

When switching between threads (a *context switch*), the CPU must save the register state of one thread and restore that of another.  
Unlike switching between processes, **the page table does not need to be changed**, since threads share the same address space.

---

### Threads and the Stack

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1755438345561/1cbece73-62ad-455d-bb69-09937b389acd.png align="center")

Each stack contains **thread-local storage**.

* This breaks the “clean” model where the heap grows upward and the stack grows downward without interference.
    
* Usually fine (stacks don’t need much space), but programs with **deep recursion** may run into stack issues.
    

---

# Why Use Threads?

Threads are mainly used for **two reasons**:

1. **Parallelism**
    
    * On multi-core systems, threads let you split work across CPUs.
        
    * Example: dividing large array operations among threads → faster execution.
        
2. **Avoiding I/O Blocking**
    
    * While one thread waits for I/O (disk, network, page fault), others can keep working.
        
    * This overlaps computation and I/O, improving responsiveness.
        

**Threads vs Processes**

* Threads share the same address space → easier data sharing.
    
* Processes are better when tasks are independent and need isolation.
    

👉 In short: **use threads for speed (parallelism) and responsiveness (I/O overlap).**

## Threads Example

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>   // sleep()

void* compute_task(void* arg) {
    for (int i = 1; i <= 5; i++) {
        printf("[Compute] step %d\n", i);
        sleep(1); // simulate a long computation
    }
    return NULL;
}

void* io_task(void* arg) {
    FILE* fp = fopen("log.txt", "w");
    if (!fp) {
        perror("fopen");
        return NULL;
    }
    for (int i = 1; i <= 5; i++) {
        fprintf(fp, "[I/O] writing line %d\n", i);
        fflush(fp); // flush immediately to disk
        printf("[I/O] wrote line %d\n", i);
        sleep(2); // simulate slow I/O
    }
    fclose(fp);
    return NULL;
}

int main() {
    pthread_t t1, t2;

    printf("Main: starting threads...\n");

    // create threads
    pthread_create(&t1, NULL, compute_task, NULL);
    pthread_create(&t2, NULL, io_task, NULL);

    // wait for both threads to finish
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("Main: all threads finished.\n");
    return 0;
}
```

result

```bash
Main: starting threads...
[Compute] step 1
[I/O] wrote line 1
[Compute] step 2
[Compute] step 3
[I/O] wrote line 2
[Compute] step 4
[Compute] step 5
[I/O] wrote line 3
[I/O] wrote line 4
[I/O] wrote line 5
Main: all threads finished.
```

🔎 Result Analysis

1. **Order is unpredictable**
    
    * Sometimes `Compute` first, sometimes `I/O` first.
        
    * Thread scheduling is **non-deterministic**.
        
2. **CPU vs I/O overlap**
    
    * `Compute`: fast, CPU-bound
        
    * `I/O`: slower, waits for disk
        
    * Shows how threads let CPU work while I/O waits.
        
3. **Key takeaway**
    
    * Threads improve efficiency by overlapping tasks.
        
    * But since outputs mix, shared data must be carefully synchronized
        

---

# Why It Gets Worse: Shared Data

You might think: *“Computers are deterministic, right? Same input, same output!”*  
But in reality, things aren’t so simple.

Here’s a classic example:

```c
static volatile int counter = 0;

void *mythread(void *arg) {
    printf("%s: begin\n", (char *) arg);
    for (int i = 0; i < 1e7; i++) {
        counter = counter + 1;
    }
    printf("%s: done\n", (char *) arg);
    return NULL;
}

int main() {
    pthread_t p1, p2;
    printf("main: begin (counter = %d)\n", counter);
    Pthread_create(&p1, NULL, mythread, "A");
    Pthread_create(&p2, NULL, mythread, "B");

    Pthread_join(p1, NULL);
    Pthread_join(p2, NULL);
    printf("main: done with both (counter = %d)\n", counter);
}
```

The intention is simple:

* Thread A and Thread B each add **1 ten million times**.
    
* The expected final result is: **20,000,000**.
    

But when you run the program, the result varies:

```c
main: done with both (counter = 20000000)   ✅ (lucky case)
main: done with both (counter = 19345221)   ❌ (wrong)
main: done with both (counter = 19221041)   ❌ (wrong)
```

It looks like a single operation in C, but the CPU breaks it down into multiple steps:

1. **Load** the value of `counter` from memory
    
2. **Add** 1
    
3. **Store** the new value back into memory
    

When multiple threads interleave these steps, things go wrong:

* Thread A: reads counter = 100
    
* Thread B: reads counter = 100
    
* Thread A: adds 1 → writes 101
    
* Thread B: adds 1 → writes 101 (**overwrites A’s update**)
    

So instead of incrementing twice, the counter only increased once.

This phenomenon is called a **Race Condition**.

# Uncontrolled Scheduling

## Increment Isn’t Atomic

Consider a simple counter increment:

```c
counter = counter + 1;
```

t the C level it looks like one operation, but at the CPU level it breaks into **three steps**:

1. Load `counter` from memory into a register
    
2. Add `1` to the register
    
3. Store the register value back into memory
    

In x86 assembly, it might look like this:

```c
mov 0x8049a1c, %eax   ; load counter
add $0x1, %eax        ; add 1
mov %eax, 0x8049a1c   ; store result
```

## How Scheduling Breaks It

Now imagine two threads running this code at the same time:

1. **Thread 1** loads `counter = 50` into its register.
    
2. A **timer interrupt** occurs — the OS saves Thread 1’s state.
    
3. **Thread 2** runs, also loads `counter = 50`, increments it, and stores `51` back to memory.
    
4. The OS switches back to **Thread 1**, which still has `eax = 51`, and then stores `51` again.
    

➡ The result is `counter = 51` instead of `52`. One increment is **lost**.

---

## Race Condition

This phenomenon is called a **race condition (data race)**:

* The final outcome depends on the exact timing of thread execution.
    
* Sometimes you get the correct result, sometimes not.
    
* Each run may give a different result → the program becomes **indeterminate**, not deterministic.
    

---

## Critical Sections

The code sequence that updates `counter` is a **critical section**:

* A region of code that accesses shared resources
    
* Must not be executed by more than one thread at a time
    

The property we need is **mutual exclusion**:

* Only one thread can be in the critical section at once
    
* Prevents lost updates and nondeterminism
    

---

Because the OS scheduler can interrupt threads at any time, **we cannot rely on “simple” code being safe in multithreaded environments**. Even a one-line increment can break.

This is the heart of concurrency problems: **uncontrolled scheduling creates race conditions**.