---
title: "Semaphores"
datePublished: Mon Aug 18 2025 11:00:44 GMT+0000 (Coordinated Universal Time)
cuid: cmeh07n4x001102lbfjbl2q2v
slug: semaphores
tags: semapore

---

**The Crux: How To Use Semaphores**

* How can semaphores be used instead of locks and condition variables?
    
* What is the definition of a semaphore?
    
* What is a binary semaphore?
    
* Is it straightforward to implement a semaphore using locks and condition variables?
    
* Conversely, is it possible to implement locks and condition variables using semaphores?
    

---

# Semaphore Summary

## 1️⃣ Definition

* A **semaphore** is a synchronization object with an integer value.
    
* Two operations:
    
    * `sem_wait()`: Decrements the value by 1; if result &lt; 0, the thread blocks.
        
    * `sem_post()`: Increments the value by 1; if threads are waiting, one is woken up.
        
* The **initial value** determines its behavior.
    

---

## 2️⃣ Types

* **Counting Semaphore**
    
    * Initial value = N → Up to N threads can access concurrently.
        
* **Binary Semaphore (= Lock)**
    
    * Initial value = 1 → Only one thread can access at a time (similar to a mutex).
        

---

## 3️⃣ Properties

* When the value is **negative**, its absolute value equals the number of waiting threads.
    
* `sem_wait()` / `sem_post()` operations are **atomic**.
    
* Semaphores are typically implemented using **locks + condition variables** internally.
    

---

## 4️⃣ Relationship

* **Semaphores from locks + condition variables** → relatively simple to implement.
    
* **Locks/condition variables from semaphores** → also possible (Dijkstra’s approach).
    
* Therefore, semaphores are a more **general-purpose synchronization primitive**.
    

---

✅ One-line summary:  
Semaphores control resource access with `wait`/`post`; depending on initialization, they can act as counting or binary semaphores. Locks and condition variables can be built from semaphores, and vice versa.

# Semaphores For Ordering

Semaphores can be used not only for mutual exclusion but also to **enforce ordering between events in concurrent programs**.  
Typical usage: one thread waits for an event, and another thread makes the event happen and signals it. The waiting thread is then awakened. In this way, the semaphore acts as an **ordering primitive**, much like condition variables.

### Answer: Initialize to 0

Two possible execution orders explain why:

#### Case 1: Parent waits before child posts

1. Parent creates child, but child has not run yet.
    
2. Parent calls `sem_wait()`. Since the semaphore is **0**, it decrements to -1 and the parent sleeps.
    
3. Child runs and calls `sem_post()`, incrementing the value back to 0 and waking the parent.
    
4. Parent resumes and finishes execution.
    

#### Case 2: Child posts before parent waits

1. Parent creates child, and the child runs first.
    
2. Child calls `sem_post()`, changing the semaphore value from 0 → 1.
    
3. Later, the parent calls `sem_wait()`. Because the value is 1, it decrements to 0 and returns immediately.
    
4. Parent does not block and the correct ordering is still preserved.
    

Got it 👍 Here’s the same summary in **English**:

---

# ✅ Producer / Consumer Problem Summary

1. **Scenario**
    
    * Producer: puts data into the buffer
        
    * Consumer: takes data out of the buffer
        
    * The buffer has a **limited size (MAX)**
        
2. **Semaphore Roles**
    
    * `empty` (initial value = MAX): number of empty slots
        
    * `full` (initial value = 0): number of filled slots
        
    * `mutex` (initial value = 1): prevents conflicts during put()/get()
        
3. **Execution Flow**
    
    * Producer:
        
        ```cpp
        empty--;   // check for an empty slot
        mutex--;   // enter critical section
        put();     // insert data into buffer
        mutex++;   // leave critical section
        full++;    // signal that one item is available
        ```
        
    * Consumer:
        
        ```cpp
        full--;    // check if data is available
        mutex--;   // enter critical section
        get();     // remove data from buffer
        mutex++;   // leave critical section
        empty++;   // signal that one slot is free
        ```
        
4. **Problem & Solution**
    
    * If `mutex` is used incorrectly → **deadlock** (threads wait forever)
        
    * Correct approach: keep `empty/full` waits **outside** the critical section,  
        and protect only `put()/get()` (the buffer access) with `mutex`.
        

---

👉 One-line summary:  
`empty/full` semaphores = “resource count management”, `mutex` = “race condition prevention”.  
All three are needed for a safe multi-producer/multi-consumer solution.