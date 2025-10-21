#include "fb.h"
#include "kprintf.h"
#include "serial.h"
#include "memory.h"
#include <stdint.h>   
#include <efi.h>  

#define COLOR_DARK_GRAY 0x00101010
extern struct font_desc font_vga_8x16;

static void delay(volatile unsigned long long count) {
    while (count--) { __asm__ __volatile__("nop"); }
}

// 해시 32바이트를 보기 좋게 찍는 헬퍼(선택)
static void print_hash(BootInfo* bi){
    kprintf(bi, "[HASH] ");
    for (int i = 0; i < 32; i++) kprintf(bi, "%02x", bi->kernel_hash[i]);
    kprintf(bi, "\n");
}

void kernel_main(BootInfo* bi) 
{
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

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

    // ✅ 새 기능: 신뢰사슬/메모리 리포트
    kprintf(bi, "[KERNEL] verified=%u\n", (unsigned)bi->verified);
    print_hash(bi);                 // 선택: 해시 헥사 출력
    memmap_report(bi);              // ✅ 메모리맵 리포트 호출


    // (데모 속도↑ 원하면 줄이기)
    delay(1000000000ULL); 

  

    while (1) { __asm__ __volatile__("hlt"); }
}
