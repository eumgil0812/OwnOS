#ifndef FONT_H
#define FONT_H

#define FONTDATAMAX (256*16)

struct font_desc {
    const char *name;
    unsigned width, height, count;
    const void *data;
};

extern const struct font_desc font_vga_8x8, font_vga_8x14;
extern unsigned char fontdata_8x16[FONTDATAMAX];
extern struct font_desc font_vga_8x16;

#endif // FONT_H
