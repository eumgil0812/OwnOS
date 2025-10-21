#ifndef FB_H
#define FB_H

#include <stdint.h>

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t  verified;
    uint8_t  kernel_hash[32];

    // NEW: ExitBootServices 이후에도 커널이 읽을 수 있게 사본 전달
    void*  MemoryMap;        // 커널이 읽을 사본 주소
    uint64_t  MemoryMapSize;    // 바이트 단위
    uint64_t  DescriptorSize;
    uint32_t ABI_Version;      // 호환성용 (예: 1)
} BootInfo;

// 📌 선언 추가
void putpixel(BootInfo* bi, int x, int y, uint32_t color);

void draw_char(BootInfo* bi, int x, int y, char c, uint32_t fg, uint32_t bg);
void fb_scroll(BootInfo* bi);
void kputs_fb(BootInfo* bi, const char* s);

#endif
