#include <stdint.h>
#include "serial.h"
#include "fb.h"
#include "font.h"

// 커서 위치 전역 변수
#define COLOR_DARK_GRAY     0x00101010


static int cursor_x = 0;
static int cursor_y = 0;
static uint32_t text_fg = 0x00FFFFFF;
static uint32_t text_bg = COLOR_DARK_GRAY;
static uint32_t screen_bg = COLOR_DARK_GRAY;

// 📜 개행 처리 함수
static void fb_newline(BootInfo* bi) {
    cursor_x = 0;
    cursor_y += font_vga_8x16.height;

    // 화면 넘어가면 맨 위로 리셋 (스크롤 기능 대신 초기화)
    if (cursor_y + font_vga_8x16.height > (int)bi->VerticalResolution) {
        cursor_y = 0;
    }
}

// 🖨️ 텍스트 로그 출력 함수
void kputs_fb(BootInfo* bi, const char* s) {
    while (*s) {
        if (*s == '\n') {
            fb_newline(bi);
        } else {
            draw_char(bi, cursor_x, cursor_y, *s, text_fg, text_bg);
            cursor_x += font_vga_8x16.width;

            // 줄 끝까지 가면 자동 개행
            if (cursor_x + font_vga_8x16.width > (int)bi->HorizontalResolution) {
                fb_newline(bi);
            }
        }
        s++;
    }
}


void kernel_main(BootInfo* bi) 
{
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    // 🟦 화면 전체 파란색으로 초기화
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    

    for (unsigned int y = 0; y < bi->VerticalResolution; y++) {
        for (unsigned int x = 0; x < bi->HorizontalResolution; x++) {
            fb[y * bi->PixelsPerScanLine + x] = screen_bg;
        }
    }
    kputs("[KERNEL] Screen cleared\n");

    // 📝 텍스트 로그 출력
    kputs_fb(bi, "[KERNEL] Boot sequence start\n");
    kputs_fb(bi, "[KERNEL] Initializing memory manager...\n");
    kputs_fb(bi, "[KERNEL] Initializing interrupt controller...\n");
    kputs_fb(bi, "[KERNEL] Initializing framebuffer console...\n");
    kputs_fb(bi, "[KERNEL] Ready.\n");

    while (1) { __asm__ __volatile__("hlt"); }
}

