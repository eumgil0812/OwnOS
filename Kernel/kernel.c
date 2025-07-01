#include <stdint.h>  // 표준 정수 타입 포함

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
} FrameBufferInfo;

void kernel_main(FrameBufferInfo* fbInfo) {
    uint32_t* fb = (uint32_t*)fbInfo->FrameBufferBase;
    uint32_t color = 0x00FF00FF;  // ARGB: Magenta


    while (1) {
    for (unsigned int y = 0; y < fbInfo->VerticalResolution; y++) {
        for (unsigned int x = 0; x < fbInfo->HorizontalResolution; x++) {
            fb[y * fbInfo->PixelsPerScanLine + x] = color;
        }
    }
}
}
