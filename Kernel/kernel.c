#include "fb.h"
#include "kprintf.h"
#include "serial.h"
#include "memory.h"
#include "pmm.h"
#include "vmm.h"
#include "idt.h"
#include "isr.h"
#include <stdint.h>
#include <efi.h>

#define COLOR_DARK_GRAY 0x00101010
extern struct font_desc font_vga_8x16;
#define PTE_W (1ULL << 1)
BootInfo* g_bootinfo = 0;  // Global BootInfo pointer

static void delay(volatile unsigned long long count) {
    while (count--) { __asm__ __volatile__("nop"); }
}

// Print 32-byte kernel hash for verification
static void print_hash(BootInfo* bi) {
    kprintf(bi, "[HASH] ");
    for (int i = 0; i < 32; i++) kprintf(bi, "%02x", bi->kernel_hash[i]);
    kprintf(bi, "\n");
}

// Boot banner display
static void bootinfo_banner(BootInfo* bi) {
    kputs_fb(bi, "===============================================================\n");
    kputs_fb(bi, "                   Tiny x86_64 Kernel Boot                     \n");
    kputs_fb(bi, "===============================================================\n");
    kprintf(bi, "ABI v%u | FB=%p %ux%u px/scan=%u | verified=%u\n",
        (unsigned)bi->ABI_Version,
        bi->FrameBufferBase,
        bi->HorizontalResolution, bi->VerticalResolution,
        bi->PixelsPerScanLine,
        (unsigned)bi->verified);
}

void kernel_main(BootInfo* bi) 
{
    g_bootinfo = bi; 
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    bootinfo_banner(bi);  // Display boot info and framebuffer details

    // Clear screen
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    for (unsigned int y = 0; y < bi->VerticalResolution; y++) {
        for (unsigned int x = 0; x < bi->HorizontalResolution; x++) {
            fb[y * bi->PixelsPerScanLine + x] = COLOR_DARK_GRAY;
        }
    }
    kputs("[KERNEL] Screen cleared\n");

    // Framebuffer console test
    kputs_fb(bi, "Enter the Kernel.\n");
    kputs_fb(bi, "[KERNEL] Framebuffer console ready.\n");

    // Trust chain and memory summary
    kprintf(bi, "[KERNEL] verified=%u\n", (unsigned)bi->verified);
    print_hash(bi);
    memmap_summary(bi);
    memmap_dump(bi, 24);

    // ------------------------------
    // PMM Initialization & Test
    // ------------------------------
    kputs("[PMM] initializing...\n");
    kputs("[PMM] Searching for largest free region...\n");
    pmm_init(bi, 0x200000ULL);
    kputs("[PMM] initialization complete\n");

    kprintf(bi, "[PMM] total=%llu pages (", (unsigned long long)pmm_total_pages());
    print_size_auto(bi, pages_to_bytes(pmm_total_pages())); kputs_fb(bi, ")\n");
    kprintf(bi, "[PMM] used =%llu pages (", (unsigned long long)pmm_used_pages());
    print_size_auto(bi, pages_to_bytes(pmm_used_pages()));  kputs_fb(bi, ")\n");

    // Test allocation: allocate 3 pages, print addresses, and free one
    void* a = pmm_alloc_page();
    void* b = pmm_alloc_page();
    void* c = pmm_alloc_page();
    kprintf(bi, "[PMM] alloc a=%p b=%p c=%p\n", a, b, c);
    kprintf(bi, "[PMM] used=%llu / %llu pages\n",
            (unsigned long long)pmm_used_pages(),
            (unsigned long long)pmm_total_pages());
    pmm_free_page(b);
    kprintf(bi, "[PMM] free b, used=%llu pages\n",
            (unsigned long long)pmm_used_pages());

    // ------------------------------
    // VMM Setup (Identity Mapping + Framebuffer)
    // ------------------------------
    kputs("[VMM] setting up page tables...\n");
    vmm_init(bi, 1);
    kputs("[VMM] done.\n");

    // 🔍 Display CPU control register states (paging + protection bits)
    uint64_t cr0, cr4;
    __asm__ volatile("mov %%cr0, %0" : "=r"(cr0));
    __asm__ volatile("mov %%cr4, %0" : "=r"(cr4));
    kprintf(bi, "[CPU] CR0=0x%llx (PG=%d, WP=%d), CR4=0x%llx (PAE=%d, PGE=%d)\n",
            cr0, !!(cr0 & (1ULL << 31)), !!(cr0 & (1ULL << 16)),
            cr4, !!(cr4 & (1ULL << 5)), !!(cr4 & (1ULL << 7)));
    // ------------------------------
    // VMM 동적 매핑 테스트
    // ------------------------------
    kputs("[VMM] dynamic mapping test...\n");

    // 1. 새 물리 페이지 확보
    void* phys = pmm_alloc_page();
    kprintf(bi, "[VMM] new phys page = %p\n", phys);

    // 2. 가상주소 지정 (예: 커널 상위 주소 영역 중 여유 주소)
    uint64_t vaddr = 0xFFFF800000100000ULL;

    // 3. 매핑
    vmm_map_page(vaddr, (uint64_t)phys, PTE_W);
    kprintf(bi, "[VMM] mapped phys=%p -> virt=0x%llx\n", phys, (unsigned long long)vaddr);


    // ------------------------------
    // VMM Mapping Verify Test (Identity 구간에서)
    // ------------------------------
    uint64_t* virt_ptr = (uint64_t*)0x00100000;   // 1 MiB 부근, 이미 아이덴티티 매핑됨
    *virt_ptr = 0xDEADBEEFCAFEBABEULL;
        
        kprintf(bi, "[VMM] wrote value 0x%llx to virt=%p\n",
            (unsigned long long)*virt_ptr, virt_ptr);
        
    uint64_t val = *virt_ptr;
        kprintf(bi, "[VMM] read value 0x%llx from virt=%p\n",
            (unsigned long long)val, virt_ptr);


    // ------------------------------
    // Enable interrupts and halt
    // ------------------------------
    interrupts_init(); 
    kputs("[IDT] interrupts enabled, halting...\n");

    while (1) { __asm__ __volatile__("hlt"); }
}
