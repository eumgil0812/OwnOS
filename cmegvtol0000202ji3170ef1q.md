---
title: "std::span"
datePublished: Mon Aug 18 2025 08:57:54 GMT+0000 (Coordinated Universal Time)
cuid: cmegvtol0000202ji3170ef1q
slug: stdspan
tags: span

---

## 1️⃣ What is `std::span`?

* `std::span` is a **view over a contiguous memory region** (such as arrays, vectors, or buffers).
    
* It does **not own the data** – it simply acts as a lightweight *slice*.
    
* If you’ve used **Python slicing** or **Rust slices**, the concept will feel familiar.
    

In short, `std::span` lets you work with arrays, vectors, or buffers **without copying them**, while providing a safe and generic interface.

---

## 2️⃣ Why do we need it?

Before C++20, code often looked like this:

1. **Manual size tracking for arrays**
    
    ```cpp
    void process(int* arr, size_t n); // must always pass length separately
    ```
    
2. **Template overload duplication**
    
    ```cpp
    void f(std::vector<int>& v);
    void f(std::array<int, 10>& arr);
    void f(int* arr, size_t n);
    ```
    
    → Each container type needed its own overload.
    
3. **Missing range information with raw pointers**  
    Passing only a pointer doesn’t tell you how many elements are valid – a common source of bugs.
    

👉 `std::span` solves all of this:

* Works with arrays, `std::vector`, `std::array`, and raw pointers
    
* Always carries size information
    
* Enables safe iteration with range-based `for`
    

---

## 3️⃣ Basic Usage

### Arrays

```cpp
#include <span>
#include <iostream>

void print(std::span<int> s) {
    for (int x : s) std::cout << x << " ";
    std::cout << "\n";
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    print(arr); // automatically converted to span
}
```

### `std::vector`

```cpp
std::vector<int> v = {10, 20, 30};
print(v); // vectors work seamlessly
```

### Slicing

```cpp
int arr[5] = {1, 2, 3, 4, 5};
std::span<int> s(arr);
auto sub = s.subspan(1, 3); // {2, 3, 4}
print(sub);
```

---

## 4️⃣ `std::span` vs `std::vector` vs Pointers

| Type | Owns Data | Stores Size | Supports Slicing | Typical Use Case |
| --- | --- | --- | --- | --- |
| `int*` | ❌ No | ❌ No | ❌ No | Low-level, C-style code |
| `std::vector` | ✅ Yes | ✅ Yes | ✅ Yes | Data storage + management |
| `std::span` | ❌ No | ✅ Yes | ✅ Yes | **Safe, generic view** |

👉 Keep the actual data in a vector (or array), but pass it around using a span for maximum clarity.

---

## 5️⃣ Practical Examples

### (1) String processing

```cpp
void print_str(std::span<const char> s) {
    for (char c : s) std::cout << c;
    std::cout << "\n";
}

int main() {
    const char* msg = "Hello";
    print_str({msg, 5}); // specify the range explicitly
}
```

### (2) Generic normalization

```cpp
template <typename T>
void normalize(std::span<T> data) {
    T max_val = *std::max_element(data.begin(), data.end());
    for (T& x : data) x /= max_val;
}
```

👉 Works with vectors, arrays, and even subspans.

---

## 6️⃣ Things to Watch Out For

* `std::span` does **not** own the data → if the underlying container is destroyed, the span becomes dangling.
    
* Be careful with temporaries:
    

```cpp
std::span<int> danger() {
    std::vector<int> v = {1, 2, 3};
    return v; // ❌ v is destroyed, span is invalid
}
```

---

## 7️⃣ Conclusion

* `std::span` is essentially a **pointer + size wrapper** introduced in C++20.
    
* It improves **safety, flexibility, and readability**.
    
* Using `std::span<>` instead of `std::vector&` in function signatures gives you **a safer and more generic interface**.
    

👉 Think of it as **the missing link between raw pointers and full containers** – a zero-overhead, range-safe view.

---