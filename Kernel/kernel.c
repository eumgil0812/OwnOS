#include "fb.h"
#include "kprintf.h"
#include "serial.h"


// 커서 위치 전역 변수
#define COLOR_DARK_GRAY     0x00101010

extern struct font_desc font_vga_8x16;


static void delay(volatile unsigned long long count) {
    while (count--) {
        __asm__ __volatile__("nop");
    }
}


void kernel_main(BootInfo* bi) 
{
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    // 🟦 화면 전체 초기화
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
    kputs_fb(bi, "[KERNEL] Checking Scroll.\n");
    
    // ⏸️ 잠시 쉬기 (눈으로 확인할 시간 주기)
    delay(1000000000ULL);

    for (int i = 1; i <=300; i++) {
        kprintf(bi, "Log line %d\n", i);
    }

    while (1) { __asm__ __volatile__("hlt"); }
    while (1) { __asm__ __volatile__("hlt"); }
}
