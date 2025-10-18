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
} BootInfo;

// 픽셀 찍기
void putpixel(BootInfo* bi, int x, int y, uint32_t color);

#endif
