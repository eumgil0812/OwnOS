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

BootInfo* g_bootinfo = 0;  // ← 전역 '정의' (유일하게 한 번만)

static void delay(volatile unsigned long long count) {
    while (count--) { __asm__ __volatile__("nop"); }
}

// 해시 32바이트를 보기 좋게 찍는 헬퍼(선택)
static void print_hash(BootInfo* bi){
    kprintf(bi, "[HASH] ");
    for (int i = 0; i < 32; i++) kprintf(bi, "%02x", bi->kernel_hash[i]);
    kprintf(bi, "\n");
}

static void bootinfo_banner(BootInfo* bi) {
    kputs_fb(bi, "===============================================================\n");
    kputs_fb(bi, "                   Tiny x86_64 Kernel Boot                     \n");
    kputs_fb(bi, "===============================================================\n");
    kprintf(bi,   "ABI v%u | FB=%p %ux%u px/scan=%u | verified=%u\n",
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

    bootinfo_banner(bi);       // 🔵 부트 배너 + 기본 정보

    // 화면 초기화
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    for (unsigned int y = 0; y < bi->VerticalResolution; y++) {
        for (unsigned int x = 0; x < bi->HorizontalResolution; x++) {
            fb[y * bi->PixelsPerScanLine + x] = COLOR_DARK_GRAY;
        }
    }
    kputs("[KERNEL] Screen cleared\n");

    // 프레임버퍼 콘솔 테스트
    kputs_fb(bi, "Enter the Kernel.\n");
    kputs_fb(bi, "[KERNEL] Framebuffer console ready.\n");

    // 신뢰사슬/메모리 리포트
    kprintf(bi, "[KERNEL] verified=%u\n", (unsigned)bi->verified);
    print_hash(bi);                 //
    memmap_summary(bi);             //
    memmap_dump(bi, 24);            // 

/*
    kputs("[PMM] initializing...\n");
    pmm_init(bi, 0x200000ULL);
    kprintf(bi, "[PMM] total=%llu pages (",
            (unsigned long long)pmm_total_pages());
    print_size_auto(bi, pages_to_bytes(pmm_total_pages())); kputs_fb(bi, ")\n");
    kprintf(bi, "[PMM] used =%llu pages (",
            (unsigned long long)pmm_used_pages());
    print_size_auto(bi, pages_to_bytes(pmm_used_pages()));  kputs_fb(bi, ")\n");

    // 0~1GiB 아이덴티티 매핑 + FB 매핑, CR3 로드 
    kputs("[VMM] setting up page tables...\n");
    vmm_init(bi, 1);
    kputs("[VMM] done.\n");

    // 테스트: 페이지 3개 할당 → 주소/통계 출력 → 일부 반납
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

    interrupts_init(); 
    kputs("[IDT] interrupts enabled, halting...\n");
*/
    while (1) { __asm__ __volatile__("hlt"); }
}

