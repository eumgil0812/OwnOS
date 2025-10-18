#include <stdint.h>
#include "fb.h"
#include "kprintf.h"
#include "serial.h"

#define COLOR_DARK_GRAY 0x00101010

extern struct font_desc font_vga_8x16;

// ============================================
// 🧱 UEFI Memory Type Definitions (UEFI 2.9)
// ============================================
enum {
    EfiReservedMemoryType        = 0,
    EfiLoaderCode                = 1,
    EfiLoaderData                = 2,
    EfiBootServicesCode          = 3,
    EfiBootServicesData          = 4,
    EfiRuntimeServicesCode       = 5,
    EfiRuntimeServicesData       = 6,
    EfiConventionalMemory        = 7,
    EfiUnusableMemory            = 8,
    EfiACPIReclaimMemory         = 9,
    EfiACPIMemoryNVS             = 10,
    EfiMemoryMappedIO            = 11,
    EfiMemoryMappedIOPortSpace   = 12,
    EfiPalCode                   = 13
};

// ============================================
// 📄 EFI Memory Descriptor Structure
// ============================================
typedef struct {
    uint32_t Type;
    uint64_t PhysicalStart;
    uint64_t VirtualStart;
    uint64_t NumberOfPages;
    uint64_t Attribute;
} EFI_MEMORY_DESCRIPTOR;

// ============================================
// 📌 Convert memory type to human-readable text
// ============================================
static const char* EfiMemoryTypeToStr(uint32_t Type) {
    switch (Type) {
        case EfiReservedMemoryType:       return "Reserved";
        case EfiLoaderCode:               return "LoaderCode";
        case EfiLoaderData:               return "LoaderData";
        case EfiBootServicesCode:         return "BS_Code";
        case EfiBootServicesData:         return "BS_Data";
        case EfiRuntimeServicesCode:      return "RT_Code";
        case EfiRuntimeServicesData:      return "RT_Data";
        case EfiConventionalMemory:       return "Conventional";
        case EfiUnusableMemory:           return "Unusable";
        case EfiACPIReclaimMemory:        return "ACPI_Reclaim";
        case EfiACPIMemoryNVS:            return "ACPI_NVS";
        case EfiMemoryMappedIO:           return "MMIO";
        case EfiMemoryMappedIOPortSpace:  return "MMIO_Port";
        case EfiPalCode:                  return "PalCode";
        default:                          return "Unknown";
    }
}

// ============================================
// 🧭 Print UEFI Memory Map
// ============================================
void print_memory_map(BootInfo* bi) {
    uint8_t* map_ptr = (uint8_t*)bi->MemoryMap;
    uint8_t* map_end = map_ptr + bi->MemoryMapSize;
    const uint64_t desc_size = bi->DescriptorSize;
    int index = 0;

    kputs_fb(bi, "\n=== [UEFI Usable Memory Map] ===\n");

    while (map_ptr < map_end) {
        EFI_MEMORY_DESCRIPTOR* desc = (EFI_MEMORY_DESCRIPTOR*)map_ptr;

        // ✅ Usable Memory만 출력
        if (desc->Type == EfiConventionalMemory) {
            uint64_t size_in_bytes = desc->NumberOfPages * 4096ULL;
            double size_in_mb = (double)size_in_bytes / (1024.0 * 1024.0);

            kprintf(bi,
                "[%02d] Start=0x%llx Pages=%llu (%.2f MB)\n",
                index,
                desc->PhysicalStart,
                desc->NumberOfPages,
                size_in_mb
            );
            index++;
        }

        map_ptr += desc_size;
    }

    if (index == 0) {
        kputs_fb(bi, "No usable memory regions found!\n");
    }

    kputs_fb(bi, "=== [End of Usable Memory Map] ===\n");
}


// ============================================
// 🧽 Framebuffer Helper
// ============================================
static void clear_screen(BootInfo* bi, uint32_t color) {
    uint32_t* fb = (uint32_t*)bi->FrameBufferBase;
    unsigned int width = bi->HorizontalResolution;
    unsigned int height = bi->VerticalResolution;
    unsigned int pitch = bi->PixelsPerScanLine;

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            fb[y * pitch + x] = color;
        }
    }
}

// ============================================
// 🕓 Simple Delay
// ============================================
static void delay(volatile unsigned long long count) {
    while (count--) {
        __asm__ __volatile__("nop");
    }
}

// ============================================
// 🧠 Kernel Entry
// ============================================
void kernel_main(BootInfo* bi) 
{
    // 🛰 Serial init
    serial_init();
    kputs("[KERNEL] Serial initialized\n");

    // 🖼 Framebuffer init
    clear_screen(bi, COLOR_DARK_GRAY);
    kputs_fb(bi, "[KERNEL] Framebuffer console ready.\n");

    // 🧭 Print UEFI Memory Map
    kputs_fb(bi, "[KERNEL] Dumping UEFI Memory Map...\n");
    print_memory_map(bi);
    kputs_fb(bi, "[KERNEL] Memory Map End.\n");

    while (1) {
        __asm__ __volatile__("hlt");
    }
}
