#ifndef EFI_MEMORY_H
#define EFI_MEMORY_H

#include <stdint.h>

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

#endif
