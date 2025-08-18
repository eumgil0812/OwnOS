---
title: "Condition Variables"
datePublished: Mon Aug 18 2025 10:26:33 GMT+0000 (Coordinated Universal Time)
cuid: cmegyzoex000202lb6qhrb45h
slug: condition-variables
tags: condition-variables

---

## Condition Variables

So far, we have studied the concept of **locks**, and we have seen that with the right combination of hardware and OS support, locks can be implemented correctly.  
However, **locks alone are not sufficient to build fully concurrent programs**.**.**

---

### **Waiting for a Condition**

In many cases, a thread must wait until a certain **condition** is satisfied before it can continue execution.  
For example:

* A parent thread may want to check whether a child thread has finished.  
    (This is commonly called a **join()**.)
    

So then, how should such waiting be implemented?

### Example (Parent Waiting For Its Child)

```c
void *child(void *arg) {
    printf("child\n");
    // XXX 여기서 "완료"를 어떻게 표시하지?
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("parent: begin\n");
    pthread_t c;
    Pthread_create(&c, NULL, child, NULL); // 자식 생성
    // XXX 부모는 어떻게 기다리지?
    printf("parent: end\n");
    return 0;
}
```

**what we want:**

```c
parent: begin
child
parent: end
```

---

### **Attempt 1: Using a Shared Variable**

```c
volatile int done = 0;

void *child(void *arg) {
    printf("child\n");
    done = 1;  // 완료 표시
    return NULL;
}

int main(int argc, char *argv[]) {
    printf("parent: begin\n");
    pthread_t c;
    Pthread_create(&c, NULL, child, NULL); // 자식 생성
    while (done == 0)
        ; // spin (계속 확인)
    printf("parent: end\n");
    return 0;
}
```

This approach generally works.  
However, the problem is:

The parent thread keeps looping until `done == 1`, wasting CPU cycles.

In other words, it’s a busy-wait approach → highly inefficient.

---

### The Crux: How To Wait For A Condition

In multi-threaded programs,  
the ability for a thread to **“wait until a certain condition becomes true”** is frequently needed.

However:

Simply spinning (looping to check repeatedly) wastes CPU cycles severely.

In some cases, it may even behave incorrectly.

👉 Thus, the key question is:  
**“How should a thread wait until a condition becomes true?”**

---

# ✅ Condition Variable

## 1\. Concept

**Mutex**: Ensures that only one thread at a time can enter the critical section (= mutual exclusion).

**Condition Variable**: Goes beyond exclusion and supports  
👉 *“waiting until a certain condition is satisfied.”*  
In other words, when one thread changes the state, another thread can wait for that state and then wake up.

**Analogy:**

* Mutex = a lock on a room so that only one person can enter at a time.
    
* Condition Variable = once inside the room, you can say *“wake me up when dinner is ready”* and fall asleep until notified.
    

---

### 2\. Relevant Functions

* `pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m)`
    
    * Sleeps until the condition is satisfied.
        
    * Must be called while holding `m`.
        
    * Internally, it **atomically releases** `m` and puts the thread to sleep.
        
    * When woken up, it re-acquires `m` before returning.
        
* `pthread_cond_signal(pthread_cond_t* c)`  
    Wakes up one waiting thread.
    
* `pthread_cond_broadcast(pthread_cond_t* c)`  
    Wakes up *all* waiting threads.
    

---

### 3\. Why is it needed? (Why `if == 1` doesn’t work)

Simply checking a flag like `if (done == 1)` fails because:

🚨 **Problem 1: Timing**  
If the parent thread checks `if (done == 0)` and is about to sleep, but at that exact moment the child sets `done = 1` and signals, the parent *misses the signal* and may sleep forever. → *classic race condition*.

🚨 **Problem 2: Busy Waiting**  
If you write `while (done == 0);`, the thread spins endlessly, consuming 100% CPU just to poll the variable. → *very inefficient*.

👉 Therefore, a plain flag cannot correctly implement “waiting for a condition.”

---

## 4\. 올바른 패턴

```c
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  c = PTHREAD_COND_INITIALIZER;
int done = 0;

void thr_exit() {
    pthread_mutex_lock(&m);
    done = 1;
    pthread_cond_signal(&c);  // 기다리는 스레드 깨우기
    pthread_mutex_unlock(&m);
}

void thr_join() {
    pthread_mutex_lock(&m);
    while (done == 0)          // 반드시 while 사용 (spurious wakeup 대비)
        pthread_cond_wait(&c, &m);
    pthread_mutex_unlock(&m);
}
```

---

## 5\. Core Rules

* **Always use with a mutex**
    
    * `wait()` must be called while holding the mutex.
        
    * Internally, it atomically releases the mutex and puts the thread to sleep.
        
* **Always check inside a** `while()` loop
    
    * Even after being woken up, the thread must re-check the condition.
        
    * Prevents *spurious wakeups* and ensures correctness.
        
* **Signal while holding the mutex**
    
    * Although in simple examples it might seem unnecessary, holding the mutex during `signal()` avoids race conditions and is strongly recommended.
        

---

### 6\. Summary

* Using only a flag like `if (==1)` cannot guarantee proper synchronization → leads to race conditions.
    
* Condition variables provide *efficient and safe waiting* until a condition is satisfied.
    
* ✅ Correct pattern = `while + wait + mutex + signal`.
    

# ✅ Producer–Consumer Problem

## 1\. Problem Definition

* **Producer**: creates data and puts it into the buffer.
    
* **Consumer**: takes data out of the buffer and uses it.
    

**Constraints**:

* If the buffer is full → the producer must wait.
    
* If the buffer is empty → the consumer must wait.
    

👉 Without synchronization, race conditions occur → incorrect behavior.

---

## 2\. Initial (Incorrect) Approach

* Using `if` conditions + a **single condition variable**
    

**Problems**:

* `if` vs `while`
    
    * Most OSes implement **Mesa semantics**: `signal()` is only a *hint* that the condition *may* have changed.
        
    * Even after being signaled, the condition might still not hold → therefore, always re-check inside a `while` loop.
        
* **Only one condition variable**
    
    * Example: a consumer calls `signal()` after consuming, but another consumer wakes up instead.
        
    * The buffer is still empty → all threads go back to sleep → **deadlock**.
        

---

## 3\. Improved Solution

* Use **two condition variables**:
    
    * `empty`: signals the producer when the buffer has space.
        
    * `fill`: signals the consumer when the buffer has data.
        

**Protocol**:

* Producer: `while (count == MAX) wait(empty)`
    
* Consumer: `while (count == 0) wait(fill)`
    
* When awakened:
    
    * Producer signals `fill` (to wake a consumer).
        
    * Consumer signals `empty` (to wake a producer).
        

👉 This prevents deadlock and ensures proper synchronization.

---

### 4\. Final Solution: Multiple-Slot Buffer

* Instead of a single buffer, extend it to **N slots** (a circular queue).
    
* Functions:
    
    ```c
    void put(int value) {
        buffer[fill_ptr] = value;
        fill_ptr = (fill_ptr + 1) % MAX;
        count++;
    }
    
    int get() {
        int tmp = buffer[use_ptr];
        use_ptr = (use_ptr + 1) % MAX;
        count--;
        return tmp;
    }
    ```
    

# ✅ Covering Condition

## 1\. Situation

Multiple threads are waiting on a condition variable `c`.

**Example: Memory Allocator**

* `allocate(size)`: succeeds only if `bytesLeft >= size`.
    
* `free(size)`: increases available memory and then calls `signal()`.
    

**Problem:** Which thread should be woken up?

Example scenario:

* `Ta`: requests `allocate(100)` → waits
    
* `Tb`: requests `allocate(10)` → waits
    
* `Tc`: calls `free(50)` → executes `signal()`
    

👉 If `signal()` wakes up `Ta`:

* Condition `bytesLeft >= 100` is still **not satisfied**, so `Ta` goes back to sleep.
    
* Meanwhile, `Tb` could have run immediately, but missed its chance → inefficiency.
    

---

### 2\. Solution: Broadcast

Instead of `pthread_cond_signal()`, use `pthread_cond_broadcast()`.

* Wake up **all waiting threads**.
    
* Each awakened thread acquires the mutex and re-checks the condition in a `while` loop.
    
* Only threads whose condition is satisfied continue; others go back to sleep.
    

✅ Ensures the “right” thread (`Tb`) proceeds.  
❌ The “wrong” thread (`Ta`) will just go back to waiting.

---

### 3\. Pros and Cons

**Pros**

* Simple and safe: no need to figure out which thread should be woken.
    
* Implements a **covering condition**: covers all possibilities.
    

**Cons**

* Wakes unnecessary threads → extra context switches.
    
* If many threads are waiting, can lead to performance degradation.
    

---

### 4\. General Guidelines

* **Use** `signal` when it’s clear which thread should proceed (e.g., producer/consumer with separate `empty` and `fill` conditions).
    
* **Use** `broadcast` when it’s ambiguous which thread can actually make progress (e.g., memory allocator, where request sizes differ).
    

👉 Rule of Thumb:  
If replacing `signal` with `broadcast` “fixes” your program, it usually means a **bug** in condition design.  
But in problems like memory allocators, where **covering conditions** are needed, `broadcast` is the correct solution.

---

Covering Condition = when you cannot know which thread to wake, use `broadcast` to wake them all, and let condition re-checking filter out who can actually proceed.

---