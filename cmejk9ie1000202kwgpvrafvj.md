---
title: "GDB Debugger"
datePublished: Wed Aug 20 2025 05:57:36 GMT+0000 (Coordinated Universal Time)
cuid: cmejk9ie1000202kwgpvrafvj
slug: gdb-debugger
tags: gdb

---

# 🐞 Introduction to GDB (GNU Debugger)

## 1\. What is GDB?

**GDB (GNU Debugger)** is a powerful tool that allows you to:

* Run your program step by step
    
* Pause at specific lines or functions (breakpoints)
    
* Inspect variables and memory in real-time
    
* Trace the call stack when something goes wrong
    

In short, GDB helps you **understand how your program is running under the hood** and catch bugs more effectively.

---

## 2\. Hands-on: Debugging `example.cpp` with GDB

### 0) Source

```cpp
// example.cpp
#include <iostream>
using namespace std;

int add(int a, int b) {
    int c = a + b;
    return c;
}

int main() {
    int x = 5, y = 10;
    int result = add(x, y);
    cout << "Result: " << result << endl;
    return 0;
}
```

### 1) Build with debug symbols

```bash
g++ -g example.cpp -o example
```

* `-g` adds line/variable info for GDB.
    

### 2) Start GDB and set a breakpoint

```bash
gdb ./example
(gdb) break add            # Stop at the start of add()
Breakpoint 1 at 0x...: file example.cpp, line 5.
```

### 3) Run until the breakpoint hits

```cpp
(gdb) run
Starting program: /path/to/example
Breakpoint 1, add (a=5, b=10) at example.cpp:5
5       int c = a + b;
```

### 4) Inspect variables

```cpp
(gdb) print a
$1 = 5
(gdb) print b
$2 = 10
```

### 5) Step over and check the new value

```cpp
(gdb) next                 # executes 'int c = a + b;'
6       return c;
(gdb) print c
$3 = 15
```

### 6) See where we came from (call stack)

```cpp
(gdb) backtrace
#0  add (a=5, b=10) at example.cpp:6
#1  0x... in main() at example.cpp:12
```

### 7) Finish the function and continue

```cpp
(gdb) finish               # run until add() returns
Run till exit from #0  add (a=5, b=10) at example.cpp:6
0x... in main() at example.cpp:12
(gdb) continue
Result: 15
[Inferior 1 (process ...) exited normally]
```

### 8) Quit GDB

```cpp
(gdb) quit
```

## 3\. Basic GDB Commands

### Starting GDB

```bash
gdb ./example
```

### Running the Program

```bash
(gdb) run
(gdb) run arg1 arg2    # Run with arguments
```

### Breakpoints

```bash
(gdb) break main       # Stop at main()
(gdb) break 15         # Stop at line 15
(gdb) delete           # Remove all breakpoints
(gdb) info breakpoints # Show current breakpoints
```

### Controlling Execution

```bash
(gdb) next     # Execute next line (skip into functions)
(gdb) step     # Step into a function
(gdb) continue # Run until the next breakpoint
(gdb) finish   # Run until the current function returns
```

### Inspecting Variables

```bash
(gdb) print x         # Print value of variable x
(gdb) print arr[3]    # Print array element
(gdb) display x       # Automatically show x after each step
(gdb) watch x         # Pause when x changes
```

### Program State

```bash
(gdb) backtrace   # Show call stack
(gdb) info locals # Show local variables
(gdb) info args   # Show function arguments
```

### Exiting GDB

```bash
(gdb) quit
```

## 4\. Why Use GDB?

* Find **logic errors** that print statements can’t show
    
* Debug **segmentation faults** by checking where the crash happened
    
* Inspect **call stacks** in recursive or complex programs
    
* Essential foundation for advanced debugging tools (like `cuda-gdb` for GPU code)
    

---