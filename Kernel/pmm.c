#include "pmm.h"
#include "memory.h"   // EFI_MEMORY_DESCRIPTOR 정의 (Type==7)
#include "kprintf.h"  // 출력용
#include <string.h>   // memset

static pmm_state_t PMM;

static inline void bitmap_set(uint64_t idx)   { PMM.bitmap[idx >> 3] |=  (1u << (idx & 7)); }
static inline void bitmap_clear(uint64_t idx) { PMM.bitmap[idx >> 3] &= ~(1u << (idx & 7)); }
static inline int  bitmap_test(uint64_t idx)  { return (PMM.bitmap[idx >> 3] >> (idx & 7)) & 1u; }

static void mark_range_used(uint64_t phys, uint64_t size) {
    if (size == 0) return;
    uint64_t start = phys;
    uint64_t end   = phys + size;
    if (end <= PMM.base || start >= PMM.base + PMM.length) return; // 관리범위 밖
    if (start < PMM.base) start = PMM.base;
    if (end   > PMM.base + PMM.length) end = PMM.base + PMM.length;

    uint64_t first = (PAGE_ALIGN_DOWN(start) - PMM.base) / PAGE_SIZE;
    uint64_t last  = (PAGE_ALIGN_UP(end)    - PMM.base) / PAGE_SIZE;
    if (last > PMM.total_pages) last = PMM.total_pages;

    for (uint64_t i = first; i < last; ++i) {
        if (!bitmap_test(i)) { bitmap_set(i); PMM.used_pages++; }
    }
}

void pmm_init(BootInfo* bi, uint64_t min_phys) {
    // 1) Select the largest "Conventional" (Type=7) memory block above min_phys
    uint8_t* p = (uint8_t*)bi->MemoryMap;
    uint64_t best_start = 0, best_len = 0;

    for (uint64_t off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(p + off);
        if (d->Type != 7) continue; // Only EfiConventionalMemory regions are usable
        uint64_t start = (uint64_t)d->PhysicalStart;
        uint64_t end   = start + ((uint64_t)d->NumberOfPages << 12);
        if (end <= min_phys) continue;
        if (start < min_phys) start = min_phys;  // Skip below min_phys
        uint64_t len = end - start;
        if (len > best_len) { best_len = len; best_start = start; }
    }

    // 2) Initialize PMM state
    PMM.base   = PAGE_ALIGN_UP(best_start);
    PMM.length = PAGE_ALIGN_DOWN(best_start + best_len) - PMM.base;
    PMM.total_pages = PMM.length / PAGE_SIZE;
    PMM.used_pages  = 0;

    // 3) Allocate bitmap memory
    //    The bitmap itself is stored in a static buffer for demo purposes.
    //    In a real kernel, this would come from kmalloc or a dedicated heap area.
    static uint8_t bitmap_storage[1 << 20]; // 1MB → 8,388,608 bits = covers ~32GB of RAM
    PMM.bitmap = bitmap_storage;
    PMM.bitmap_bytes = (PMM.total_pages + 7) / 8;
    if (PMM.bitmap_bytes > sizeof(bitmap_storage)) {
        // If too large, clamp it to prevent overflow (demo safety measure).
        uint64_t max_pages = (sizeof(bitmap_storage) * 8ULL);
        PMM.total_pages = (max_pages < PMM.total_pages) ? max_pages : PMM.total_pages;
        PMM.length = PMM.total_pages * PAGE_SIZE;
        PMM.bitmap_bytes = (PMM.total_pages + 7) / 8;
    }
    memset(PMM.bitmap, 0, PMM.bitmap_bytes);

    // 4) Mark reserved regions
    //    (a) Memory outside PMM range is ignored automatically
    //    (b) Known reserved resources are explicitly marked as used:
    //       - Kernel code/data (roughly 1MiB–current link end)
    //       - Framebuffer region
    //       - Bootloader’s memory map copy (optional, if known)
    //
    // The framebuffer base is assumed to be a physical address (as provided by GOP).
    // If it’s virtual, this reservation can be skipped.
    mark_range_used((uint64_t)(uintptr_t)bi->FrameBufferBase,
                    (uint64_t)bi->PixelsPerScanLine * bi->VerticalResolution * 4ULL);

    // Optionally reserve low memory (e.g., 0x100000–0x400000) used by kernel
    mark_range_used(0x00100000ULL, 0x00300000ULL);

    kprintf(bi, "[PMM] base=0x%llx, len=%llu MB, pages=%llu, bitmap=%llu bytes\n",
            (unsigned long long)PMM.base,
            (unsigned long long)(PMM.length >> 20),
            (unsigned long long)PMM.total_pages,
            (unsigned long long)PMM.bitmap_bytes);
}


void* pmm_alloc_page(void) {
    if (!PMM.bitmap || PMM.total_pages == 0) return NULL;

    // Simple linear scan
    for (uint64_t i = 0; i < PMM.total_pages; ++i) {
        if (!bitmap_test(i)) {                // Found a free page
            bitmap_set(i);                    // Mark it as used
            PMM.used_pages++;                 // Increment usage count
            return (void*)(uintptr_t)(PMM.base + i * PAGE_SIZE); // Return physical address
        }
    }
    return NULL; // No free pages available
}


void pmm_free_page(void* pa) {
    if (!pa) return;
    uint64_t phys = (uint64_t)(uintptr_t)pa;

    if (phys < PMM.base || phys >= PMM.base + PMM.length) return; // Outside managed range
    if ((phys - PMM.base) & (PAGE_SIZE - 1)) return;              // Not page-aligned

    uint64_t idx = (phys - PMM.base) / PAGE_SIZE;
    if (bitmap_test(idx)) {
        bitmap_clear(idx);
        if (PMM.used_pages) PMM.used_pages--;
    }
}


uint64_t pmm_total_pages(void)  { return PMM.total_pages; }
uint64_t pmm_used_pages(void)   { return PMM.used_pages;  }
uint64_t pmm_base_phys(void)    { return PMM.base;        }
uint64_t pmm_length_bytes(void) { return PMM.length;      }

void pmm_reserve_range(uint64_t phys, uint64_t size) {
    mark_range_used(phys, size);
}

void pmm_debug_dump(void (*print)(const char*)) {
    (void)print; // 필요 시 구현
}

