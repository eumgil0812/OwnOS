#include <stdint.h>
#include "fb.h"
#include "serial.h"
#include "font.h"   



// 📝 문자열 출력
void draw_string(BootInfo* bi, int x, int y, const char* s, uint32_t fg, uint32_t bg) {
    while (*s) {
        draw_char(bi, x, y, *s++, fg, bg);
        x += font_vga_8x16.width;
    }
}