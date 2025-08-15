---
title: "RAII - (1) Bad Case"
datePublished: Fri Aug 15 2025 05:36:33 GMT+0000 (Coordinated Universal Time)
cuid: cmeceb73p000802k13nwbfpt6
slug: raii-1-bad-case
tags: raii

---

# Introduction – Why RAII?

## **Resource Acquisition Is Initialization**

In modern C++ development, resource management is one of the most critical aspects of writing reliable and maintainable software. Whether you are working on high-performance computing (HPC) applications, system-level programming, or embedded systems, every resource you acquire — memory, file handles, network sockets, mutex locks, or even GPU buffers — must be released at the right time.

Manual resource management often leads to subtle bugs: memory leaks, dangling pointers, double frees, or unclosed file descriptors. These issues may remain hidden during development, only to surface under heavy load or in production, where they can cause severe performance degradation or even system crashes.

**RAII** (*Resource Acquisition Is Initialization*) addresses this problem by tying the lifetime of a resource to the lifetime of an object. With RAII, resources are acquired during object construction and automatically released during destruction, ensuring exception safety and making cleanup deterministic. This approach not only reduces boilerplate code but also makes programs more robust, especially in complex, multi-threaded, or long-running HPC workloads.

# Study With Case

## 0) Common compiler options

```bash
g++ -std=c++20 -O0 -g -fsanitize=address -fno-omit-frame-pointer file.cpp -o app
```

**g++**  
Means using the GNU C++ compiler.

**\-std=c++20**  
Compile using the C++20 standard.  
→ Allows use of modern features (e.g., `concepts`, `ranges`).

**\-O0**  
Optimization level 0 (no optimization).  
→ Keeps code structure intact, making it easier to trace variables and source lines during debugging.

**\-g**  
Include debugging information.  
→ Enables debuggers such as `gdb` or `lldb` to inspect variable values, source code lines, etc.

**\-fsanitize=address**  
Enable AddressSanitizer (ASan).  
→ Detects memory errors at runtime (buffer overflows, use-after-free, etc.).

**\-fno-omit-frame-pointer**  
Do not omit the function call stack frame pointer.  
→ Allows more accurate call stack traces during debugging.  
(With optimization enabled, frame pointers may be removed, making stack tracing harder.)

**file.cpp**  
The source file to compile.

**\-o app**  
Specify the output file name as `app`.  
→ If omitted, the default output name is `a.out`.

## 1)Memory leak

1. ```cpp
    // leak.cpp
    #include <cstdlib>
    
    int main() {
        // Allocate 100MB (intentionally large)
        size_t N = 100 * 1024 * 1024;
        char* buf = (char*)std::malloc(N);
        // ... Assume it was used ...
        // Mistake: Program ends without free(buf) → memory leak
        return 0;
    }
    ```
    
    ```bash
    g++ -std=c++20 -O0 -g leak.cpp -o leak
    valgrind --leak-check=full ./leak
    ```
    
    ```bash
    ➜  RAII g++ -std=c++20 -O0 -g leak.cpp -o leak
    valgrind --leak-check=full ./leak
    
    ==19160== Memcheck, a memory error detector
    ==19160== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.
    ==19160== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
    ==19160== Command: ./leak
    ==19160== 
    ==19160== 
    ==19160== HEAP SUMMARY:
    ==19160==     in use at exit: 104,857,600 bytes in 1 blocks
    ==19160==   total heap usage: 1 allocs, 0 frees, 104,857,600 bytes allocated
    ==19160== 
    ==19160== 104,857,600 bytes in 1 blocks are possibly lost in loss record 1 of 1
    ==19160==    at 0x4846828: malloc (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
    ==19160==    by 0x109168: main (leak.cpp:8)
    ==19160== 
    ==19160== LEAK SUMMARY:
    ==19160==    definitely lost: 0 bytes in 0 blocks
    ==19160==    indirectly lost: 0 bytes in 0 blocks
    ==19160==      possibly lost: 104,857,600 bytes in 1 blocks
    ==19160==    still reachable: 0 bytes in 0 blocks
    ==19160==         suppressed: 0 bytes in 0 blocks
    ==19160== 
    ==19160== For lists of detected and suppressed errors, rerun with: -s
    ==19160== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
    ```
    
    \==19160== possibly lost: 104,857,600 bytes in 1 blocks
    
    This means the program allocated 100 MB of memory in a single block, and by the time the program exited, Valgrind could still see a pointer to it somewhere in memory — but not in a way that guarantees safe access. In practice, this almost always indicates a memory leak: the memory was never freed (free or delete was not called), so the operating system had to reclaim it when the process ended.
    

## 2) Dangling pointer

### Example1

```cpp
// dangling_stack.cpp
#include <iostream>

int* bad() {
    int x = 42;       // exists on the stack
    return &x;        // Wrong: disappears after scope ends → dangling
}

int main() {
    int* p = bad();
    std::cout << *p << "\n"; // UB: if lucky, shows 42; usually dangerous
}
```

```bash
g++ -std=c++20 -O0 -g -fsanitize=address -fno-omit-frame-pointer dangling_stack_clobber.cpp -o ds_asan2
ASAN_OPTIONS=detect_stack_use_after_return=1 ./ds_asan2
```

```bash
dangling_stack.cpp: In function ‘int* bad()’:
dangling_stack.cpp:6:12: warning: address of local variable ‘x’ returned [-Wreturn-local-addr]
    6 |     return &x;       // Wrong: disappears after scope ends → dangling
      |            ^~
dangling_stack.cpp:5:9: note: declared here
    5 |     int x = 42;      // exists on the stack
      |         ^
AddressSanitizer:DEADLYSIGNAL
=================================================================
==22873==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000 (pc 0x57ac293bd3c7 bp 0x7ffe672cec20 sp 0x7ffe672cec10 T0)
==22873==The signal is caused by a READ memory access.
==22873==Hint: address points to the zero page.
    #0 0x57ac293bd3c7 in main /home/joy0812/study/RAII/dangling_stack.cpp:11
    #1 0x7e70d822a1c9 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #2 0x7e70d822a28a in __libc_start_main_impl ../csu/libc-start.c:360
    #3 0x57ac293bd1a4 in _start (/home/joy0812/study/RAII/ds_asan+0x11a4) (BuildId: 966a1768ba81fa2f4527dbc3e99c2fdc2039cb08)

AddressSanitizer can not provide additional info.
SUMMARY: AddressSanitizer: SEGV /home/joy0812/study/RAII/dangling_stack.cpp:11 in main
==22873==ABORTING
```

This example demonstrates a stack dangling pointer caused by returning the address of a local variable.  
Once the `bad()` function exits, `x` is removed from the stack, making the memory pointed to by `p` invalid.  
With AddressSanitizer’s `detect_stack_use_after_return=1` option enabled, the stack frame is poisoned after it goes out of scope, and any access to that memory immediately triggers an error.  
In this run, instead of accidentally printing a stale value, the program crashed with a segmentation fault right away.

### Example2

```cpp
// dangling_heap.cpp
#include <iostream>

int main() {
    int* p = new int(7);
    delete p;               // 해제
    std::cout << *p << "\n"; // ❌ use-after-free
}
```

```bash
g++ -std=c++20 -O0 -g dangling_heap.cpp -o dh_dbg
valgrind --tool=memcheck --leak-check=full --track-origins=yes ./dh_dbg
```

```bash
==23318== Memcheck, a memory error detector
==23318== Copyright (C) 2002-2022, and GNU GPL'd, by Julian Seward et al.
==23318== Using Valgrind-3.22.0 and LibVEX; rerun with -h for copyright info
==23318== Command: ./dh_dbg
==23318== 
==23318== Invalid read of size 4
==23318==    at 0x1091E3: main (dangling_heap.cpp:7)
==23318==  Address 0x4e2e080 is 0 bytes inside a block of size 4 free'd
==23318==    at 0x484A61D: operator delete(void*, unsigned long) (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==23318==    by 0x1091DE: main (dangling_heap.cpp:6)
==23318==  Block was alloc'd at
==23318==    at 0x4846FA3: operator new(unsigned long) (in /usr/libexec/valgrind/vgpreload_memcheck-amd64-linux.so)
==23318==    by 0x1091BE: main (dangling_heap.cpp:5)
==23318== 
7
==23318== 
==23318== HEAP SUMMARY:
==23318==     in use at exit: 0 bytes in 0 blocks
==23318==   total heap usage: 3 allocs, 3 frees, 74,756 bytes allocated
==23318== 
==23318== All heap blocks were freed -- no leaks are possible
==23318== 
==23318== For lists of detected and suppressed errors, rerun with: -s
==23318== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
```

look at the

```bash
==23318== Invalid read of size 4
==23318==    at 0x1091E3: main (dangling_heap.cpp:7)
==23318==  Address 0x4e2e080 is 0 bytes inside a block of size 4 free'd
```

* **Invalid read of size 4** → An invalid 4-byte read (here, the size of an `int`).
    
* **dangling\_heap.cpp:7** → The source line where the problem occurred.
    
* **free’d** → Attempting to read memory that has already been freed (**heap-use-after-free**).
    

## 3) Double Free

```cpp
  1 // double_free.cpp
  2 #include <cstdlib>
  3 
  4 int main() {
  5     int* p = (int*)std::malloc(sizeof(int));
  6     std::free(p);
  7     std::free(p); // ❌ 두 번 해제
  8 }
  9 
```

```bash
g++ -std=c++20 -O0 -g -fsanitize=address double_free.cpp -o df
./df
```

```bash
=================================================================
==23505==ERROR: AddressSanitizer: attempting double-free on 0x502000000010 in thread T0:
    #0 0x76f8956fc4d8 in free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52
    #1 0x5713f40c81da in main /home/joy0812/study/RAII/double_free.cpp:7
    #2 0x76f894e2a1c9 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58
    #3 0x76f894e2a28a in __libc_start_main_impl ../csu/libc-start.c:360
    #4 0x5713f40c80e4 in _start (/home/joy0812/study/RAII/df+0x10e4) (BuildId: 34d0247bfdd8f82e92b51f8862aff19241a366d1)
```

## 4) Unclosed FD

```cpp
  1 // fd_leak.cpp
  2 #include <cstdio>
  3 #include <stdexcept>
  4 
  5 void write_something() {
  6     FILE* fp = std::fopen("out.txt", "w"); // Acquire resource
  7     if (!fp) throw std::runtime_error("open failed");
  8 
  9     // ... Assume an error occurs in the middle ...
 10     throw std::runtime_error("something went wrong");
 11     // fclose(fp); // ❌ Not reached → not closed
 12 }
 13 
 14 int main() {
 15     try { write_something(); }
 16     catch (...) {}
 17 }
 18 
```

For short-lived programs, the OS reclaims file descriptors when the process terminates, but for long-lived processes or those with repeated loops, file descriptor exhaustion can occur and cause failures.

## 5) Mutex locks also follow the same pattern

```cpp
#include <mutex>
void f(std::mutex& m) {
    std::lock_guard<std::mutex> lock(m); // Automatically unlocks
    // ...
}
```

This code is generally a safe pattern, but the problem arises when the **lock scope** is too large or the lock is held longer than necessary.

* While the lock is held, other threads cannot access the mutex, so performing long operations (e.g., I/O, heavy computation) inside the locked region **significantly reduces parallelism**.
    
* Even though RAII ensures the lock is released if an exception is thrown, it also makes it harder to fine-tune the lock scope in exception-handling flows.
    
* In particular, **locking the entire function** makes debugging and maintenance harder and may cause severe lock contention.