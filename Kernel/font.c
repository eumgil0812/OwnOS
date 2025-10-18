#include <stdint.h>
#include "fb.h"
#include "serial.h"
#include "font.h"   


// 🖼️ 단일 문자 출력
void draw_char(BootInfo* bi, int x, int y, char c, uint32_t fg, uint32_t bg) {
    const unsigned char* glyph = font_vga_8x16.data + (c * font_vga_8x16.height);

    for (int row = 0; row < font_vga_8x16.height; row++) {
        unsigned char bits = glyph[row];
        for (int col = 0; col < font_vga_8x16.width; col++) {
            if (bits & (1 << (7 - col)))
                putpixel(bi, x + col, y + row, fg);
            else
                putpixel(bi, x + col, y + row, bg);
        }
    }
}

// 📝 문자열 출력
void draw_string(BootInfo* bi, int x, int y, const char* s, uint32_t fg, uint32_t bg) {
    while (*s) {
        draw_char(bi, x, y, *s++, fg, bg);
        x += font_vga_8x16.width;
    }
}
