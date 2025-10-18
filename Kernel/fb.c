#include "fb.h"

// 🧱 FrameBuffer 픽셀 출력
void putpixel(BootInfo* bi, int x, int y, uint32_t color) {
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    fb[y * bi->PixelsPerScanLine + x] = color;
}
