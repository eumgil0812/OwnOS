#ifndef FONT_H
#define FONT_H

#include <stdint.h>
#include "fb.h"           // BootInfo 사용

#define FONTDATAMAX (256 * 16)

struct font_desc {
    const char* name;
    unsigned width, height, count;
    const void* data;
};

extern unsigned char fontdata_8x16[FONTDATAMAX];
extern struct font_desc font_vga_8x16;


void draw_string(BootInfo* bi, int x, int y, const char* s, uint32_t fg, uint32_t bg);

#endif
