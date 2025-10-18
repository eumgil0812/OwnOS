#include <stdint.h>
#include <string.h>
#include "fb.h"
#include "font.h"

#define COLOR_DARK_GRAY 0x00101010

extern struct font_desc font_vga_8x16;

static int cursor_x = 0;
static int cursor_y = 0;
static uint32_t text_fg = 0x00FFFFFF;
static uint32_t text_bg = COLOR_DARK_GRAY;

// 📜 개행 처리 함수
static void fb_newline(BootInfo* bi) {
    cursor_x = 0;
    cursor_y += font_vga_8x16.height;
    if (cursor_y + font_vga_8x16.height > (int)bi->VerticalResolution) {
        fb_scroll(bi);
    }
}

// 🖼️ 문자 출력 함수
void draw_char(BootInfo* bi, int x, int y, char c, uint32_t fg, uint32_t bg) {
    const uint8_t* glyph = (const uint8_t*)font_vga_8x16.data + (c * font_vga_8x16.height);
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    for (int row = 0; row < (int)font_vga_8x16.height; row++) {
        uint8_t bits = glyph[row];
        for (int col = 0; col < (int)font_vga_8x16.width; col++) {
            uint32_t color = (bits & (1 << (7 - col))) ? fg : bg;
            fb[(y + row) * bi->PixelsPerScanLine + (x + col)] = color;
        }
    }
}

// 📜 스크롤 기능
void fb_scroll(BootInfo* bi) {
    int line_height = font_vga_8x16.height;
    int screen_width = bi->HorizontalResolution;
    int screen_height = bi->VerticalResolution;

    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    size_t pitch = bi->PixelsPerScanLine;

    size_t copy_bytes = (screen_height - line_height) * pitch * sizeof(uint32_t);
    memmove(
        fb,
        fb + (line_height * pitch),
        copy_bytes
    );

    for (int y = screen_height - line_height; y < screen_height; y++) {
        for (int x = 0; x < screen_width; x++) {
            fb[y * pitch + x] = text_bg;
        }
    }

    cursor_y = screen_height - line_height;
    cursor_x = 0;
}

// 🖨️ 문자열 출력 함수
void kputs_fb(BootInfo* bi, const char* s) {
    while (*s) {
        if (*s == '\n') {
            fb_newline(bi);
        } else {
            draw_char(bi, cursor_x, cursor_y, *s, text_fg, text_bg);
            cursor_x += font_vga_8x16.width;
            if (cursor_x + font_vga_8x16.width > (int)bi->HorizontalResolution) {
                fb_newline(bi);
            }
        }
        s++;
    }
}

// 📌 단일 픽셀을 그리는 기본 함수
void putpixel(BootInfo* bi, int x, int y, uint32_t color) {
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    fb[y * bi->PixelsPerScanLine + x] = color;
}
