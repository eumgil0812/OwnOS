---
title: "(5) UEFI Application  - Basic"
datePublished: Mon Jun 16 2025 11:20:33 GMT+0000 (Coordinated Universal Time)
cuid: cmbz06gan001w02ih3scp3je9
slug: 5-uefi-application-basic

---

Since I want to build a UEFI application on my own, I decided to take a course. I hope that one day, I’ll be good enough to teach someone something.

[https://youtu.be/t3iwBQg\_Gik?list=PLT7NbkyNWaqZYHNLtOZ1MNxOt8myP5K0p](https://youtu.be/t3iwBQg_Gik?list=PLT7NbkyNWaqZYHNLtOZ1MNxOt8myP5K0p)

## 🛠 Installing Essential Development Tools on Ubuntu

### ✅ Compile Tools

```bash
sudo apt update                          # Updates the local package list
sudo apt install clang                  # Installs the Clang C/C++ compiler
sudo apt install gcc                    # Installs the GNU C compiler (GCC)
sudo apt install make                   # Installs the 'make' build automation tool
sudo apt install build-essential        # Installs essential tools for building software (gcc, g++, make, etc.)
sudo apt install mingw-w64              # Installs cross-compiler for building Windows executables on Linux
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750067763145/6abe93cc-e8c1-4a29-b24d-fcae1daade83.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750067969377/89e1869b-664f-4368-aa8a-75a442ece7f2.png align="center")

## ✅ QEMU

```bash
sudo apt update                             
sudo apt install qemu qemu-system          #
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virt-manager  
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750068140739/c4ed3b5e-78e4-497a-9f48-69e79536ed39.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750068209229/95d91112-93b2-4c02-abb9-53e5886852c6.png align="center")

### ✅ What is OVMF?

**OVMF** stands for **Open Virtual Machine Firmware**.

It is an **open-source firmware** implementation that follows the **UEFI (Unified Extensible Firmware Interface)** specification.

OVMF provides a UEFI environment for virtual machines running on platforms like **QEMU**, **KVM**, and **VirtualBox**.

It allows virtual machines to boot in **UEFI mode** instead of the traditional BIOS mode.

---

### 📁 Typical Installation Path

```bash
ls /usr/share/OVMF/ # Read-only code section of the UEFI firmware
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750070272739/64f06202-92b6-49d0-b2f8-e9ec04dc06d5.png align="center")

| **Purpose** | **Filename** | **Description** |
| --- | --- | --- |
| Code | `OVMF_CODE.fd` | Read-only. Contains the executable code of the UEFI firmware. |
| Variables | `OVMF_VARS.fd` | Read/write. Stores UEFI environment variables (e.g., boot entries, configs). |

### 🧩 Why is OVMF Needed?

If you want to run a **UEFI application (.efi)** or boot a **UEFI-based OS image** (e.g., Windows or Ubuntu in UEFI mode) using QEMU,  
👉 you need actual UEFI firmware to provide the required environment.

While QEMU comes with BIOS support by default,  
**UEFI is not included by default**, so **you need to install OVMF separately** to enable UEFI functionality.

## ✅UEFI BOOTING

Copy the `OVMF_VARS_4M.fd` file into your current working directory:

```bash
cp /usr/share/OVMF/OVMF_VARS_4M.fd .
```

Then run the command again:

```bash
qemu-system-x86_64 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=OVMF_VARS_4M.fd \
  -net none
```

---

| **Option** | **Meaning** |
| --- | --- |
| `qemu-system-x86_64` | Launches a QEMU virtual machine for the x86\_64 architecture. |
| `-drive if=pflash,...,file=OVMF_CODE_4M.fd` | Attaches the UEFI firmware code in read-only mode. This is the actual firmware binary. |
| `-drive if=pflash,...,file=OVMF_VARS_4M.fd` | Attaches a writable UEFI variable storage. Used to save configuration changes. |
| `-net none` | Disables networking for the virtual machine (i.e., no network interfaces are enabled). |

Since `OVMF_VARS_4M.fd` gets modified when saving settings or registering `.efi` files in the UEFI shell, **you should not use the system version directly.** Always work with a local copy.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1750070777605/18deaa18-02f6-4131-a649-77a501a03a85.png align="center")

### ✅ What is this screen?

This is the **console environment built into UEFI firmware**, similar to what you'd find on a real computer.

Unlike BIOS systems, UEFI systems can provide an interactive shell interface **before the operating system boots**, allowing you to enter commands directly.

### 🖥 Current Situation Summary:

* You’ve successfully booted QEMU in **UEFI mode**.
    
* There’s no OS or `.efi` application loaded yet, so it boots into the **UEFI Shell** by default.
    
* The prompt `Shell>` is the command line of the UEFI Interactive Shell.
    

### 🔍 Screen Breakdown

```bash
UEFI Interactive Shell v2.2
UEFI v2.70 (Ubuntu distribution of EDK II, 0x00010000)
...
Shell>
```

* `BLK0:`: This shows the currently detected disk/device mappings.
    
* `startup.nsh`: No startup script was found, so the system pauses here.
    
* `Shell>`: You can enter UEFI shell commands at this prompt.
    

….