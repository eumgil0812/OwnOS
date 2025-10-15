#include <stdint.h>

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];
} BootInfo;

void kernel_main(BootInfo* bootInfo) {
    // 🔐 Secure Boot check
    if (bootInfo->verified != 1) {
        // Fill the screen with red and halt the system if the signature check failed
        uint32_t* fb = (uint32_t*)bootInfo->FrameBufferBase;
        for (unsigned int y = 0; y < bootInfo->VerticalResolution; y++) {
            for (unsigned int x = 0; x < bootInfo->HorizontalResolution; x++) {
                fb[y * bootInfo->PixelsPerScanLine + x] = 0x00FF0000; // Red
            }
        }
        while (1) { __asm__ __volatile__("hlt"); }
    }

    // 🟣 Normal boot (Secure Boot verification passed)
    uint32_t* fb = (uint32_t*)bootInfo->FrameBufferBase;
    uint32_t color = 0x00FF00FF;  // Magenta

    while (1) {
        for (unsigned int y = 0; y < bootInfo->VerticalResolution; y++) {
            for (unsigned int x = 0; x < bootInfo->HorizontalResolution; x++) {
                fb[y * bootInfo->PixelsPerScanLine + x] = color;
            }
        }
    }
}
