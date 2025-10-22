#pragma once
#include <stdint.h>
#include "fb.h"   


#ifndef EFI_MEMORY_RUNTIME
#define EFI_MEMORY_UC                0x0000000000000001ULL
#define EFI_MEMORY_WC                0x0000000000000002ULL
#define EFI_MEMORY_WT                0x0000000000000004ULL
#define EFI_MEMORY_WB                0x0000000000000008ULL
#define EFI_MEMORY_UCE               0x0000000000000010ULL
#define EFI_MEMORY_WP                0x0000000000001000ULL
#define EFI_MEMORY_RP                0x0000000000002000ULL
#define EFI_MEMORY_XP                0x0000000000004000ULL
#define EFI_MEMORY_NV                0x0000000000008000ULL
#define EFI_MEMORY_MORE_RELIABLE     0x0000000000010000ULL
#define EFI_MEMORY_RO                0x0000000000020000ULL
#define EFI_MEMORY_SP                0x0000000000040000ULL
#define EFI_MEMORY_RUNTIME           0x8000000000000000ULL
#endif


typedef struct {
    uint32_t Type;
    uint32_t Pad;
    uint64_t PhysicalStart;
    uint64_t VirtualStart;
    uint64_t NumberOfPages;
    uint64_t Attribute;
} __attribute__((packed)) EFI_MEMORY_DESCRIPTOR;


const char* efi_type(uint32_t type);
uint64_t pages_to_bytes(uint64_t pages);
void print_size_auto(BootInfo* bi, uint64_t bytes);
void memmap_dump(BootInfo* bi, uint32_t limit);

void memmap_summary(BootInfo* bi);

void memmap_report(BootInfo* bi);

void print_mem_attrs(BootInfo* bi, uint64_t a);