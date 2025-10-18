#include <stdint.h>
#include "font_8x16.h"

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];
} BootInfo;

// I/O 포트 함수 정의
static inline void outb(uint16_t port, uint8_t val) {
    __asm__ __volatile__("outb %0, %1" : : "a"(val), "Nd"(port));
}

static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ __volatile__("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}

// Serial 초기화
void serial_init() {
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x80);
    outb(0x3F8 + 0, 0x03);
    outb(0x3F8 + 1, 0x00);
    outb(0x3F8 + 3, 0x03);
    outb(0x3F8 + 2, 0xC7);
    outb(0x3F8 + 4, 0x0B);
}

// Serial 출력
static inline void serial_out(char c) {
    while ((inb(0x3F8 + 5) & 0x20) == 0);
    outb(0x3F8, c);
}

void kputs(const char* s) {
    while (*s) serial_out(*s++);
}

// 메인 커널
void kernel_main(BootInfo* bi) {
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    uint32_t color = 0x00FF00FF;

    for (unsigned int y = 0; y < 100; y++) {
        for (unsigned int x = 0; x < 100; x++) {
            fb[y * bi->PixelsPerScanLine + x] = color;
        }
    }

    kputs("[KERNEL] Painted 100x100\n");

    while (1) { __asm__ __volatile__("hlt"); }
}
