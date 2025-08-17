---
title: "RAII - (2) Solution with RAII"
datePublished: Sun Aug 17 2025 05:09:48 GMT+0000 (Coordinated Universal Time)
cuid: cmef88hoj000302jva77cfn1p
slug: raii-2-solution-with-raii
tags: cpp, raii

---

**RAII (Resource Acquisition Is Initialization)** is a simple but powerful idea in modern C++. The concept is this: whenever you create an object, it immediately *owns* some resource (like memory, a file handle, or a lock). And when that object goes out of scope, the resource is released automatically—no manual cleanup, no leaks, no forgotten `close()` calls.

In this post, I’ll walk through some of the most common resources we deal with in C and C++, and how RAII helps us manage them safely. We’ll cover things like dynamic memory, file descriptors, and locks, and show how RAII-based tools like `std::vector`, `std::unique_ptr`, or `std::lock_guard` make your code safer and cleaner.

---

## 1) `std::vector` — the go-to solution for dynamic memory

```cpp
// leak_fixed.cpp
#include <vector>

int main() {
    std::vector<char> buf(100 * 1024 * 1024); // automatic lifetime management
    // ... use buf ...
} // buffer is automatically released when leaving scope
```

👉 Instead of manually calling `new[]` and `delete[]`, `std::vector` takes care of allocation and cleanup for you. When `main()` returns, the vector’s destructor automatically frees the memory—no leaks, no worries.

## 2) Returning by Value / Ownership Transfer

```cpp
// dangling_fixed.cpp
#include <iostream>

int make_value() { 
    return 42; // return by value: safe
}

int main() {
    int v = make_value(); 
    std::cout << v << "\n";
}
```

## 3) Using `std::unique_ptr` for Ownership Transfer

```cpp
#include <iostream>
#include <memory>

std::unique_ptr<int> make_ptr() {
    auto p = std::make_unique<int>(42);
    return p; // ownership is transferred (move)
}

int main() {
    auto p = make_ptr();
    std::cout << *p << "\n"; // automatically freed when leaving scope
}
```

👉 Here, `std::unique_ptr` ensures that only one place owns the resource at a time. When `make_ptr()` returns, ownership of the integer is **moved** to the caller. When `p` goes out of scope in `main()`, the memory is automatically released—no manual `delete` required, no leaks, no dangling pointers.

### What is `std::unique_ptr`?

* One of the standard smart pointers in C++.
    
* The key idea: **it has exactly one owner** of the resource.
    
* Therefore, **copying is forbidden**, but **moving is allowed**.
    
* When the owner goes out of scope (the `unique_ptr` is destroyed), it automatically calls `delete` → preventing memory leaks.
    

## 4) File Stream in C++

```cpp
// file_stream.cpp
#include <fstream>

int main() {
    std::ofstream ofs("out.txt"); // Constructor automatically opens the file
    ofs << "hello\n";             // Write to the stream
} // End of scope → destructor automatically closes the file
```

What happens here?

* `std::ofstream ofs("out.txt");`  
    When the object `ofs` is created, the file `out.txt` is automatically opened.
    
* `ofs << "hello\n";`  
    Using the stream operator (`<<`), the string is written into the file.
    
* **End of scope (function exit)**  
    When `main` ends, the destructor of `ofs` is called. This automatically invokes `close()`.  
    Even if an exception or an early `return` occurs, the file is guaranteed to be closed properly.
    

---

✅ Why is this good?

* **Safer than C-style** `fopen`/`fclose`  
    No risk of forgetting to call `fclose()` manually.
    
* **Exception safety**  
    Resources are released even if an exception occurs.
    
* **Cleaner syntax**  
    Stream-based operations (`<<`, `>>`) are more intuitive and readable than low-level C APIs.
    

## 5) std::lock\_guard

```cpp
#include <mutex>
void f(std::mutex& m) {
    std::lock_guard<std::mutex> lock(m); // 예외/조기리턴에도 자동 unlock
    // ...
}
```

## 6) RAII rapper in C++

In C++, **RAII (Resource Acquisition Is Initialization)** is all about tying resource management to object lifetime.  
An **RAII wrapper** is simply a small class that "wraps" a raw resource (like memory, file handles, sockets, or mutexes) and manages its acquisition and release automatically.

Instead of calling `malloc/free` or `fopen/fclose` manually, the RAII wrapper ensures:

* Acquisition happens in the constructor
    
* Release happens in the destructor
    

This makes the code **exception-safe** and **deterministic**.

---

### Example 1: File RAII Wrapper

```cpp
#include <cstdio>
#include <stdexcept>

class FileWrapper {
    FILE* fp;
public:
    FileWrapper(const char* path, const char* mode) {
        fp = std::fopen(path, mode);
        if (!fp) throw std::runtime_error("Failed to open file");
    }
    ~FileWrapper() {
        if (fp) std::fclose(fp);
    }

    FILE* get() { return fp; }
};

int main() {
    FileWrapper file("out.txt", "w");
    std::fprintf(file.get(), "Hello RAII\n");
} // Destructor automatically calls fclose()
```

### Example 2: Memory RAII Wrapper

```cpp
#include <cstdlib>

class Buffer {
    char* data;
    size_t size;
public:
    Buffer(size_t n) : size(n) {
        data = (char*)std::malloc(size);
    }
    ~Buffer() {
        std::free(data);
    }

    char* get() { return data; }
};
```

✅ Why use RAII Wrappers?

* **Exception safety**: resources are always released, even if an exception is thrown.
    
* **Simpler code**: no need for manual cleanup or `try-finally` constructs.
    
* **Standardized approach**: many STL classes are RAII wrappers by design, e.g.,
    
    * `std::unique_ptr` for memory
        
    * `std::lock_guard` for mutexes
        
    * `std::vector` for dynamic arrays