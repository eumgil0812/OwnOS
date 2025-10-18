#ifndef FB_H
#define FB_H

#include <stdint.h>

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];

    // 📌 UEFI Memory Map 전달
    void* MemoryMap;
    uint64_t MemoryMapSize;
    uint64_t DescriptorSize;
    uint64_t DescriptorVersion;
} BootInfo;


// 📌 선언 추가
void putpixel(BootInfo* bi, int x, int y, uint32_t color);

void draw_char(BootInfo* bi, int x, int y, char c, uint32_t fg, uint32_t bg);
void fb_scroll(BootInfo* bi);
void kputs_fb(BootInfo* bi, const char* s);

#endif
